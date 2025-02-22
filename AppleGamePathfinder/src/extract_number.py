import cv2
import numpy as np
import os

# 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")

# 이미지 경로
img_path = f"{IMAGE_DIR}/full_image.png"
img_cv = cv2.imread(img_path)

# 이미지를 흑백으로 변환
img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# 엣지 감지
img_canny = cv2.Canny(img_gray, 100, 200)

# 컨투어(윤곽선) 찾기
cnts, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 추출된 숫자 영역을 담을 배열 초기화
array = []

# 각 컨투어 처리
for cnt in cnts:
    # 직사각형 경계박스를 구함
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    aspect_ratio = float(w) / h  # 가로/세로 비율

    # 직사각형의 가로/세로 비율 + 면적으로 숫자만 추출
    if (aspect_ratio > 0.2) and (aspect_ratio < 0.8) and (rect_area > 300) and (rect_area < 600):
        array.append(cnt)
        
array = list(array)

# 추출된 숫자 영역을 화면에 표시
cv2.imshow("Extracted Number Areas", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 추출된 숫자 영역을 저장할 디렉토리 생성
EXTRACT_DIR= os.path.join(IMAGE_DIR, "extract")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# 숫자 영역을 잘라서 이미지로 저장
for idx, cnt in enumerate(array):
    # 숫자 영역을 추출
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img_cv[y:y + h, x:x + w]
    
    # 추출된 숫자 영역을 파일로 저장
    roi_path = f"{IMAGE_DIR}/extract/extract{idx+1}.png"
    cv2.imwrite(roi_path, roi)
    print(f"Saved extracted number image to {roi_path}")
