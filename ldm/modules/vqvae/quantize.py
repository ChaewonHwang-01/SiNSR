import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def _chunked_argmin_dist(z_flattened: torch.Tensor, emb: torch.Tensor, chunk_size: int):
    """
    z_flattened: (N, D)
    emb: (K, D)
    return indices: (N,)
    OOM-safe: computes distances in chunks.
    """
    # norms
    emb_norm = torch.sum(emb ** 2, dim=1)                     # (K,)
    z_norm = torch.sum(z_flattened ** 2, dim=1, keepdim=True) # (N,1)

    all_indices = []
    N = z_flattened.shape[0]
    for i in range(0, N, chunk_size):
        zc = z_flattened[i:i+chunk_size]      # (c, D)
        zc_norm = z_norm[i:i+chunk_size]      # (c, 1)
        # (c, K)
        d = zc_norm + emb_norm.unsqueeze(0) - 2.0 * (zc @ emb.t())
        all_indices.append(torch.argmin(d, dim=1))
    return torch.cat(all_indices, dim=0)


class VectorQuantizer2(nn.Module):
    """
    VectorQuantizer2 (the one your autoencoder imports):
      from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

    This version keeps the original interface but patches distance computation to be OOM-safe.
    """
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
        chunk_size=256,   # <-- OOM-safe knob
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.chunk_size = int(chunk_size)

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape, device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # keep interface compatibility
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"

        # z: (B,C,H,W) -> (B,H,W,C)
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)  # (N,D)

        # -------- OOM-safe argmin distance --------
        emb = self.embedding.weight  # (K,D)
        min_encoding_indices = _chunked_argmin_dist(z_flattened, emb, self.chunk_size)  # (N,)
        # -----------------------------------------

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        perplexity = None
        min_encodings = None

        # loss
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # straight-through
        z_q = z + (z_q - z).detach()

        # back to (B,C,H,W)
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        # remap logic (kept)
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        if indices.dim() == 2 and indices.shape[1] == 1:
            indices = indices[:, 0]
        indices = indices.long()

        z_q = self.embedding(indices)

        if shape is not None:
            # shape: (B,H,W,C)
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


# Optional alias if other parts expect VectorQuantizer
VectorQuantizer = VectorQuantizer2
