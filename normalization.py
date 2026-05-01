import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm  # 진행 상황 표시용

def normalize_images_with_clipping(input_dir, output_dir, clip_min=0, clip_max=255, clip_percentile=False):
    """
    이미지 데이터셋에 정규화와 클리핑을 적용하여 새로운 데이터셋 생성
    
    Args:
        input_dir (str): 원본 이미지가 있는 디렉토리 경로
        output_dir (str): 정규화된 이미지가 저장될 디렉토리 경로
        clip_min (float): 클리핑할 최소값 (기본값: 0)
        clip_max (float): 클리핑할 최대값 (기본값: 255)
        clip_percentile (bool): True일 경우 백분위수 기반 클리핑 사용 (1%와 99%)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # PNG 파일 찾기
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    if not png_files:
        print(f"경고: {input_dir}에 PNG 파일이 없습니다.")
        return
    
    print(f"총 {len(png_files)}개의 PNG 파일을 정규화 및 클리핑합니다...")
    
    # 각 이미지에 대해 처리
    for img_path in tqdm(png_files):
        # 파일 이름 추출
        filename = os.path.basename(img_path)
        
        try:
            # 이미지 로드
            img = Image.open(img_path)
            
            # 이미지를 NumPy 배열로 변환
            img_array = np.array(img).astype(np.float32)
            
            # 이미지 모드에 따라 정규화 및 클리핑 적용
            if img.mode == "RGB" or img.mode == "RGBA":
                # 각 채널별로 분리
                channels = []
                
                # RGB 또는 RGBA 이미지의 경우 각 채널별로 정규화
                for c in range(3):  # RGB 채널만 처리
                    channel = img_array[..., c]
                    # 실제 데이터 부분만 찾기 (0이 아닌 값들)
                    valid_pixels = channel[channel > 0]
                    
                    if len(valid_pixels) > 0:
                        # 백분위수 기반 클리핑 적용 (선택 사항)
                        if clip_percentile:
                            p_low, p_high = np.percentile(valid_pixels, [1, 99])
                            # 백분위수 기반으로 클리핑 (정규화 전)
                            channel = np.clip(channel, p_low, p_high)
                            valid_pixels = channel[channel > 0]
                        
                        # 실제 데이터 부분의 최소값과 최대값 계산
                        min_val = np.min(valid_pixels)
                        max_val = np.max(valid_pixels)
                        
                        # 정규화: (x - min) / (max - min)
                        if max_val > min_val:  # 0으로 나누는 것 방지
                            normalized = (channel - min_val) / (max_val - min_val)
                            # 0-255 범위로 변환
                            normalized = normalized * 255
                            # 클리핑 적용 (정규화 후)
                            normalized = np.clip(normalized, clip_min, clip_max)
                        else:
                            normalized = channel  # 변화 없음
                        
                        channels.append(normalized)
                    else:
                        channels.append(channel)  # 변화 없음
                
                # 알파 채널이 있는 경우 그대로 유지
                if img.mode == "RGBA":
                    channels.append(img_array[..., 3])  # 알파 채널 추가
                    
                # 채널 합치기
                normalized_img_array = np.stack(channels, axis=-1).astype(np.uint8)
                normalized_img = Image.fromarray(normalized_img_array)
            else:
                # 그레이스케일 이미지의 경우
                valid_pixels = img_array[img_array > 0]
                
                if len(valid_pixels) > 0:
                    # 백분위수 기반 클리핑 적용 (선택 사항)
                    if clip_percentile:
                        p_low, p_high = np.percentile(valid_pixels, [1, 99])
                        # 백분위수 기반으로 클리핑 (정규화 전)
                        img_array = np.clip(img_array, p_low, p_high)
                        valid_pixels = img_array[img_array > 0]
                    
                    min_val = np.min(valid_pixels)
                    max_val = np.max(valid_pixels)
                    
                    if max_val > min_val:
                        normalized_img_array = (img_array - min_val) / (max_val - min_val) * 255
                        # 클리핑 적용 (정규화 후)
                        normalized_img_array = np.clip(normalized_img_array, clip_min, clip_max)
                        normalized_img_array = normalized_img_array.astype(np.uint8)
                        normalized_img = Image.fromarray(normalized_img_array)
                    else:
                        normalized_img = img  # 변화 없음
                else:
                    normalized_img = img  # 변화 없음
            
            # 정규화된 이미지 저장 - 원본 크기와 동일하게 유지
            normalized_img.save(os.path.join(output_dir, filename))
            
        except Exception as e:
            print(f"오류 발생: {filename} - {str(e)}")
    
    print(f"모든 이미지 정규화 및 클리핑 완료! 처리된 이미지 수: {len(png_files)}")

# 실행 예시
if __name__ == "__main__":
    # 입력 및 출력 디렉토리 설정
    input_dir = "dataset/dataset_1/mite/"  # 원본 이미지가 있는 디렉토리
    output_dir = "dataset/dataset_1_normalized/mite/"  # 정규화된 이미지가 저장될 디렉토리
    
    # 정규화 및 클리핑 실행 - 일반 클리핑 (0-255)
    normalize_images_with_clipping(input_dir, output_dir)
    
    # 또는 백분위수 기반 클리핑 사용
    # normalize_images_with_clipping(input_dir, output_dir, clip_percentile=True)
    
    # 또는 사용자 정의 클리핑 범위 사용
    # normalize_images_with_clipping(input_dir, output_dir, clip_min=10, clip_max=240)