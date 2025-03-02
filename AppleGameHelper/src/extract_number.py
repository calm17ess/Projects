import cv2
import os
import numpy as np
import pytesseract
from skimage.metrics import structural_similarity as ssim

# 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")

# 추출된 숫자 영역을 저장할 디렉토리 생성
EXTRACT_DIR = os.path.join(IMAGE_DIR, "numbers")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# 이미지 경로
img_path = f"{IMAGE_DIR}/full_image.png"
img_cv = cv2.imread(img_path)  # 컬러 이미지를 그대로 로드

# 이미지를 흑백으로 변환
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# 엣지 감지
img_canny = cv2.Canny(img_gray, 100, 200)

# 컨투어(윤곽선) 찾기
cnts, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 추출된 숫자 영역을 담을 배열 초기화
img_rect = img_cv.copy()
array = []

# rect_area 동적 설정
img_area = img_gray.shape[0] * img_gray.shape[1]

img_area_min = img_area * 0.0004
img_area_max = img_area * 0.0008

# 각 컨투어 처리
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    aspect_ratio = float(w)/ h

    if (0.2 < aspect_ratio < 0.8) and (img_area_min < rect_area < img_area_max):
        array.append(tuple(cnt.reshape(-1)))
        cv2.rectangle(img_rect, (x, y), (x + w, y + h), (255, 0, 0), 1)

array = list(set(array))
print(f"array: {len(array)}, {array}")



# 중복된 이미지를 확인할 set
unique_images = []
threshold = 0.9  # 유사도 threshold (0.98 이상의 유사도는 중복으로 간주)

# 첫 번째 이미지의 크기 저장 (기준 크기)
reference_size = None

# 중복 제거 및 저장
for idx, cnt in enumerate(array):
    x, y, w, h = cv2.boundingRect(np.array(cnt).reshape(-1, 1, 2))
    roi = img_cv[y-1:y + h+1, x-1:x + w+1]  # 컬러 이미지를 그대로 사용

    # 첫 번째 이미지의 크기를 기준으로 모든 이미지를 리사이즈
    if reference_size is None:
        reference_size = (w, h)  # 첫 번째 이미지의 크기 저장
    roi_resized = cv2.resize(roi, reference_size)  # 기준 크기로 리사이즈

    # 이미지에서 숫자 추출
    extracted_text = pytesseract.image_to_string(roi_resized, config='--psm 10 --oem 3')  # 숫자만 추출하도록 설정
    extracted_text = ''.join(filter(str.isdigit, extracted_text))  # 숫자만 필터링

    # 추출된 텍스트가 비어있지 않으면 저장
    if extracted_text:
        # 파일명에 숫자 사용
        roi_path = os.path.join(EXTRACT_DIR, f"{extracted_text}.png")
        
        # 중복 제거 및 저장
        is_duplicate = False
        for unique_img in unique_images:
            similarity = ssim(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY), cv2.cvtColor(unique_img, cv2.COLOR_BGR2GRAY))
            
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(roi_resized)
            cv2.imwrite(roi_path, roi_resized)  # 숫자에 맞는 파일명으로 저장
            print(f"Saved extracted number image as {roi_path}")
    else:
        print(f"No text detected in region {idx}, skipping.")


