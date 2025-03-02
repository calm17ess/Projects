# Copyright (c) 2025 calm17ess
# Licensed under the MIT License. See LICENSE file for details.

import cv2
import numpy as np
import pyautogui
import time
import os
import sys
import pytesseract
from skimage.metrics import structural_similarity as ssim

# 타이틀 텍스트
SelectROIText = "Drag your Applegame outline"
MainTitleText = "Apple Game Helper"

threshold = 0.8

def get_resource_path(relative_path):
    # PyInstaller 실행 파일에서도 리소스 경로를 올바르게 찾는 함수
    if getattr(sys, 'frozen', False):  # EXE로 실행 중인지 확인
        base_path = sys._MEIPASS  # PyInstaller의 임시 디렉터리
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))  # 프로젝트 루트 (src의 상위 폴더)

    return os.path.join(base_path, relative_path)

IMAGE_DIR = get_resource_path("Images")

# 마우스 이벤트 전역 변수
roi = None  # 선택된 영역 (x, y, w, h)
dragging = False
start_x, start_y = -1, -1
frame = None

# numbers 폴더가 존재하면 삭제
EXTRACT_DIR = os.path.join(IMAGE_DIR, "numbers")
if os.path.exists(EXTRACT_DIR):
    import shutil
    shutil.rmtree(EXTRACT_DIR)  # 폴더 삭제

# 폴더 새로 생성
os.makedirs(EXTRACT_DIR, exist_ok=True)

def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, roi, dragging, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        temp_img = frame.copy()
        cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow(SelectROIText, temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        w, h = abs(x - start_x), abs(y - start_y)
        if w > 0 and h > 0:
            roi = (min(start_x, x), min(start_y, y), w, h)
        dragging = False

# 초기 화면 캡처 및 ROI(처리 수행 영역) 선택
frame = np.array(pyautogui.screenshot())
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow(SelectROIText, frame)
cv2.setMouseCallback(SelectROIText, mouse_callback)

while roi is None:
    cv2.waitKey(1)

cv2.destroyWindow(SelectROIText)

x, y, w, h = roi
if w <= 0 or h <= 0:
    print("영역이 잘못되었습니다. 다시 실행해주세요.")
    exit()

# 이미지 경로
img_cv = frame[y:y+h, x:x+w]  # 컬러 이미지를 그대로 로드

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
    x2, y2, w2, h2 = cv2.boundingRect(cnt)
    rect_area = w2 * h2
    aspect_ratio = float(w2)/ h2

    if (0.2 < aspect_ratio < 0.8) and (img_area_min < rect_area < img_area_max):
        array.append(tuple(cnt.reshape(-1)))
        cv2.rectangle(img_rect, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 1)
array = list(set(array))

# 중복된 이미지를 확인할 set
unique_images = []
image_threshold = 0.86  # 유사도 threshold 

# 첫 번째 이미지의 크기 저장 (기준 크기)
reference_size = None
duplicate_arr = []

# 중복 제거 및 저장
for idx, cnt in enumerate(array):
    x2, y2, w2, h2 = cv2.boundingRect(np.array(cnt).reshape(-1, 1, 2))
    roi = img_cv[y2-1:y2+1 + h2, x2-1:x2 + w2+1]
    # 첫 번째 이미지의 크기를 기준으로 모든 이미지를 리사이즈
    if reference_size is None:
        reference_size = (w2, h2)  # 첫 번째 이미지의 크기 저장
    roi_resized = cv2.resize(roi, reference_size)  # 기준 크기로 리사이즈

    # 이미지에서 숫자 추출
    extracted_text = pytesseract.image_to_string(roi_resized, config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')  # 숫자만 추출하도록 설정
    extracted_text = ''.join(filter(str.isdigit, extracted_text))  # 숫자만 필터링

    # 추출된 텍스트가 비어있지 않으면 저장
    if extracted_text:
        # 파일명에 숫자 사용
        roi_path = os.path.join(EXTRACT_DIR, f"{extracted_text}.png")
        
        # 중복 제거 및 저장
        is_duplicate = False
        for unique_img in unique_images:
            similarity = ssim(cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY), cv2.cvtColor(unique_img, cv2.COLOR_BGR2GRAY))
            
            if similarity >= image_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(roi_resized)
            cv2.imwrite(roi_path, roi_resized)  # 숫자에 맞는 파일명으로 저장
            print(f"Saved extracted number image as {roi_path}")
            duplicate_arr.append(extracted_text)
        else:
            print(f"No text detected in region {idx}, skipping.")
    
    if len(duplicate_arr) == 9: break;

# 이미지 로드
digit_images = {}
for i in range(1, 10):
    img_path = os.path.join(EXTRACT_DIR, f"{i}.png")
    digit_images[i] = cv2.imread(img_path, 0)
    if digit_images[i] is None:
        print(f"이미지 로드 실패: {img_path}")

time.sleep(1)

# 실시간 감지 루프
while True:
    screenshot = np.array(pyautogui.screenshot())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cropped = screenshot[y:y+h, x:x+w]
    img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    apple_array = np.zeros((10, 17), dtype=int)
    for digit, digit_img in digit_images.items():
        res = cv2.matchTemplate(img_gray, digit_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            row, col = int(pt[1] / (h / 10)), int(pt[0] / (w / 17))
            if 0 <= row < 10 and 0 <= col < 17:
                apple_array[row, col] = digit

    def find_valid_sums(arr):
        result = []
        rows, cols = arr.shape

        # 가로 합 검사
        for i in range(rows):
            for j in range(cols):
                for k in range(j+1, cols+1):
                    sub_arr = arr[i, j:k]
                    if np.sum(sub_arr) == 10 and sub_arr[0] != 0 and sub_arr[-1] != 0:
                        result.append(((i, j), (i, k-1)))

        # 세로 합 검사
        for i in range(cols):
            for j in range(rows):
                for k in range(j+1, rows+1):
                    sub_arr = arr[j:k, i]
                    if np.sum(sub_arr) == 10 and sub_arr[0] != 0 and sub_arr[-1] != 0:
                        result.append(((j, i), (k-1, i)))

        return result    

    result = find_valid_sums(apple_array)

    used_positions = set()
    if result:
        for start, end in result:
            if start in used_positions or end in used_positions:
                continue

            center_x1 = int(start[1] * (w / 17) + (w / 34) + x)
            center_y1 = int(start[0] * (h / 10) + (h / 20) + y)
            center_x2 = int(end[1] * (w / 17) + (w / 34) + x)
            center_y2 = int(end[0] * (h / 10) + (h / 20) + y)

            thickness = max(1, min(w, h) // 100) # 선 두께를 이미지 크기에 따라 조정
            color = (0, 0, 0)
            cv2.line(cropped, (center_x1 - x, center_y1 - y), (center_x2 - x, center_y2 - y), color, thickness)

            used_positions.add(start)
            used_positions.add(end)

    # 숫자 개수 세기
    digit_counts = {i: np.sum(apple_array == i) for i in range(1, 10)}

    # 하단 추가 영역
    additional_height = 40  # 추가 영역의 높이를 늘려서 숫자 표시 공간을 만듭니다
    result_with_space = np.vstack((cropped, np.zeros((additional_height, cropped.shape[1], 3), dtype=np.uint8)))

    # 숫자 개수를 일렬로 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = max(0.5, min(w, h) / 900)  # 글자 크기를 이미지 크기에 따라 조정
    x_offset = 8  # 왼쪽에서 시작
    y_offset = cropped.shape[0] + 30  # 하단 위치 조정

    # 일렬로 표시하기 위한 간격 설정
    spacing = max(50, w//10) # 숫자 간격을 설정

    for digit, count in digit_counts.items():
        cv2.putText(result_with_space, f"{digit}:{count} ", (x_offset, y_offset), font, fontScale, (255, 255, 255), 1, cv2.LINE_AA)
        x_offset += spacing  # 숫자 간격을 두고 이동

    # 결과 이미지 표시
    cv2.imshow(MainTitleText, result_with_space)
    # ESC나 창의 버튼을 눌러 종료
    if cv2.getWindowProperty(MainTitleText, cv2.WND_PROP_VISIBLE) < 1: break
    if cv2.waitKey(1) & 0xFF == 27: break


cv2.destroyAllWindows()
