import os
import glob
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm  # 진행 상황 표시용

def resize_images_with_safeguards(input_dir, output_dir1, output_dir2, target_size=(224, 224), 
                                 filter_type=Image.LANCZOS, pre_blur=False, post_clip=True):
    """
    이미지 데이터셋을 안전하게 리사이징하는 함수
    
    Args:
        input_dir (str): 입력 이미지 디렉토리
        output_dir1 (str): 단순 리사이즈된 이미지 저장 디렉토리
        output_dir2 (str): 패딩된 이미지 저장 디렉토리
        target_size (tuple): 목표 이미지 크기 (가로, 세로)
        filter_type: 리사이징 필터 (LANCZOS, BICUBIC, BILINEAR 등)
        pre_blur (bool): 리사이징 전 약간의 블러 적용 여부
        post_clip (bool): 리사이징 후 클리핑 적용 여부
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    
    # PNG 파일 찾기
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    if not png_files:
        print(f"경고: {input_dir}에 PNG 파일이 없습니다.")
        return
    
    print(f"총 {len(png_files)}개의 PNG 파일을 처리합니다...")
    
    # 각 이미지에 대해 처리
    for img_path in tqdm(png_files):
        # 파일 이름 추출
        filename = os.path.basename(img_path)
        
        try:
            # 이미지 로드
            img = Image.open(img_path)
            
            # 선택적으로 약간의 블러 적용 (경계 효과 감소)
            if pre_blur:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # 1. 단순 리사이징 - 이미지를 NumPy 배열로 변환하여 처리
            img_resized = img.resize(target_size, filter_type)
            
            # 리사이징 후 클리핑 적용 (선택 사항)
            if post_clip:
                img_resized_arr = np.array(img_resized)
                if img.mode == "RGB" or img.mode == "RGBA":
                    # RGB 채널만 클리핑
                    img_resized_arr[..., :3] = np.clip(img_resized_arr[..., :3], 0, 255)
                else:
                    # 그레이스케일 클리핑
                    img_resized_arr = np.clip(img_resized_arr, 0, 255)
                
                img_resized = Image.fromarray(img_resized_arr)
            
            # 저장
            img_resized.save(os.path.join(output_dir1, filename))
            
            # 2. 종횡비 유지하며 패딩 추가
            # 원본 이미지 크기
            width, height = img.size
            
            # 종횡비 유지를 위한 비율 계산
            ratio = min(target_size[0] / width, target_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 종횡비 유지하며 리사이즈
            img_ratio_resized = img.resize((new_width, new_height), filter_type)
            
            # 리사이징 후 클리핑 적용 (선택 사항)
            if post_clip:
                img_ratio_arr = np.array(img_ratio_resized)
                if img.mode == "RGB" or img.mode == "RGBA":
                    # RGB 채널만 클리핑
                    img_ratio_arr[..., :3] = np.clip(img_ratio_arr[..., :3], 0, 255)
                else:
                    # 그레이스케일 클리핑
                    img_ratio_arr = np.clip(img_ratio_arr, 0, 255)
                
                img_ratio_resized = Image.fromarray(img_ratio_arr)
            
            # 패딩을 위한 새 이미지 생성 (검은색 배경)
            padded_img = Image.new("RGBA" if img.mode == "RGBA" else "RGB", 
                                  target_size, 
                                  (0, 0, 0, 0) if img.mode == "RGBA" else (0, 0, 0))
            
            # 중앙에 리사이즈된 이미지 붙이기
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            padded_img.paste(img_ratio_resized, (paste_x, paste_y))
            
            # 저장
            padded_img.save(os.path.join(output_dir2, filename))
            
        except Exception as e:
            print(f"오류 발생: {filename} - {str(e)}")
    
    print(f"모든 이미지 처리 완료! 처리된 이미지 수: {len(png_files)}")

# 실행 예시
if __name__ == "__main__":
    # 입력 및 출력 디렉토리 설정
    input_dir = "dataset/dataset_2_deblurred_normalized/bee"  # 원본 이미지가 있는 디렉토리
    output_dir1 = "dataset/dataset_2_deblurred_normalized_resized/bee"  # 단순 리사이즈된 이미지가 저장될 디렉토리
    output_dir2 = "dataset/dataset_2_deblurred_normalized_resized_set/bee"  # 패딩된 이미지가 저장될 디렉토리
    
    # 기본 LANCZOS 필터 사용, 리사이징 후 클리핑 적용
    resize_images_with_safeguards(input_dir, output_dir1, output_dir2, post_clip=True)
    
    # 또는 다른 필터 사용
    # from PIL import Image
    # resize_images_with_safeguards(input_dir, output_dir1, output_dir2, filter_type=Image.BICUBIC)
    
    # 또는 블러+클리핑 모두 적용
    # resize_images_with_safeguards(input_dir, output_dir1, output_dir2, pre_blur=True, post_clip=True)