import os
import cv2

def resize_images_from_txt(txt_path, output_dir, size=(256, 256), overwrite=False):
    """
    txt_path : 이미지 경로들이 줄마다 저장된 txt 파일 경로
    output_dir : 리사이즈된 이미지 저장 폴더
    size : (width, height)
    overwrite : True면 원본 위에 덮어쓰기, False면 output_dir에 저장
    """

    # output 폴더 생성
    if not overwrite:
        os.makedirs(output_dir, exist_ok=True)

    # txt 읽기
    with open(txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[INFO] {len(image_paths)}개의 이미지를 처리합니다.")

    for idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"[WARN] 존재하지 않음: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 읽기 실패: {img_path}")
            continue

        resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

        # 저장 경로 설정
        if overwrite:
            save_path = img_path
        else:
            # 파일 이름만 유지 (폴더 구조 무시)
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 저장
        cv2.imwrite(save_path, resized)

        if (idx + 1) % 100 == 0 or idx == len(image_paths) - 1:
            print(f"[INFO] {idx+1}/{len(image_paths)} 처리 완료 → {save_path}")

    print("[DONE] 모든 이미지 리사이즈 완료 ✅")


if __name__ == "__main__":
    # ===== 사용자 설정 =====
    txt_path = "/home/lee/SinSR/traindata/traindataset.txt"   # 이미지 경로 리스트
    output_dir = "/home/lee/SinSR/dataset/train_resized_128"   # 저장 폴더
    target_size = (128, 128)                               # 리사이즈 크기
    overwrite_original = False                             # 원본 덮어쓰기 여부

    resize_images_from_txt(txt_path, output_dir, target_size, overwrite_original)
