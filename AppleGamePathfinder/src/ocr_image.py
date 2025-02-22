import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "Images")

# 이미지 로드
img = cv2.imread(f'{IMAGE_DIR}/full_image.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환

# 엣지 처리
edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)  # 엣지 검출
cv2.imshow("Edges", edges)

# 윤곽 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()

# 윤곽선 그리기 (검출된 윤곽선을 원본 이미지에 그리기)
array = []
for i in range(len(contours)):
    cnt = cv2.boundingRect(contours[i])
    x, y, w, h = cnt
    rect_area = w * h
    aspect_ratio = float(w) / h  # 가로/세로 비율

    if (aspect_ratio > 0.2) and (aspect_ratio < 0.8) and (rect_area > 300) and (rect_area < 600):
        array.append(cnt)
        cv2.rectangle(img_contours, (x, y), (x+w, y+h), (255, 0, 0), 2)

array = set(array)
cv2.imshow("contours", img_contours)
print(array)
print(contours)
# 숫자 템플릿 이미지 로드 (1부터 9까지)
digit_images = {}
for i in range(1, 10):
    digit_images[i] = cv2.imread(f'{IMAGE_DIR}/numbers/{i}.png', 0)

# 숫자 인식 후 좌표에 숫자 추가하기 위한 배열 생성 (10x17 배열)
apple_array = np.zeros((10, 17), dtype=int)

#  각 윤곽에 대해 템플릿 매칭을 이용해 숫자 인식
for contour in array:
    # 윤곽 영역을 바운딩 박스로 만듦
    x, y, w, h = contour
    
    # 바운딩 박스 내의 숫자 영역 잘라내기
    digit_roi = img_gray[y:y+h, x:x+w]
    
    best_match = None
    best_val = 0
    
    # 템플릿 매칭을 통해 숫자 인식
    for digit, digit_img in digit_images.items():
        res = cv2.matchTemplate(digit_roi, digit_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # 임계치 이상일 경우 가장 잘 맞는 숫자 선택
        if max_val > best_val and max_val >= 0.99: # 임계 값
            best_val = max_val
            best_match = digit
    
    # 숫자를 인식한 경우 해당 좌표에 숫자 추가
    if best_match is not None:
        # 좌표에 숫자 추가
        row, col = int(y / (img.shape[0] / 10)), int(x / (img.shape[1] / 17))  # 10x17 배율로 좌표 변환
        apple_array[row, col] = best_match

# 7. 인식된 배열 출력
print(apple_array)

# 결과 이미지에 사각형 표시
img_result = img.copy()
for contour in array:
    # 바운딩 박스 그리기
    x, y, w, h = contour
    cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 각 숫자의 좌표에 대해 점을 찍음
data = np.array(np.where(apple_array > 0))  # 숫자가 있는 위치 찾기
plt.scatter(data[1], data[0])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
