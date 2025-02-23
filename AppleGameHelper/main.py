import cv2
import numpy as np
import pyautogui
import time
import os
import sys

threshold = 0.9

def get_resource_path(relative_path):
    """ PyInstaller 실행 파일에서도 리소스 경로를 올바르게 찾는 함수 """
    if getattr(sys, 'frozen', False):  # EXE로 실행 중인지 확인
        base_path = sys._MEIPASS  # PyInstaller의 임시 디렉터리
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))  # 프로젝트 루트 (src의 상위 폴더)

    return os.path.join(base_path, relative_path)

IMAGE_DIR = get_resource_path("Images/numbers")

# 이미지 로드
digit_images = {}
for i in range(1, 10):
    img_path = os.path.join(IMAGE_DIR, f"{i}.png")
    digit_images[i] = cv2.imread(img_path, 0)
    if digit_images[i] is None:
        print(f"⚠️ 이미지 로드 실패: {img_path}")

# 마우스 이벤트를 위한 전역 변수
roi = None  # 선택된 영역 (x, y, w, h)
dragging = False
start_x, start_y = -1, -1
frame = None

def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, roi, dragging, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        temp_img = frame.copy()
        cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        w, h = abs(x - start_x), abs(y - start_y)
        if w > 0 and h > 0:
            roi = (min(start_x, x), min(start_y, y), w, h)
        dragging = False

# 초기 화면 캡처 및 ROI(처리 수행 영역) 선택
frame = np.array(pyautogui.screenshot())
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow("Select ROI", frame)
cv2.setMouseCallback("Select ROI", mouse_callback)

while roi is None:
    cv2.waitKey(1)

cv2.destroyWindow("Select ROI")

x, y, w, h = roi
if w <= 0 or h <= 0:
    print("Invalid ROI size. Exiting...")
    exit()

# 초기 배열 저장을 위한 시간 대기
time.sleep(2)

# 초기 상태 저장
screenshot = np.array(pyautogui.screenshot())
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
cropped = screenshot[y:y+h, x:x+w]
img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

first_array = np.zeros((10, 17), dtype=int)
for digit, digit_img in digit_images.items():
    res = cv2.matchTemplate(img_gray, digit_img, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        row, col = int(pt[1] / (h / 10)), int(pt[0] / (w / 17))
        if 0 <= row < 10 and 0 <= col < 17:
            first_array[row, col] = digit

# 실시간 감지 루프
while True:
    screenshot = np.array(pyautogui.screenshot())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cropped = screenshot[y:y+h, x:x+w]
    img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # 실시간 이미지 숫자 인식
    apple_array = np.zeros((10, 17), dtype=int)
    for digit, digit_img in digit_images.items():
        res = cv2.matchTemplate(img_gray, digit_img, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            row, col = int(pt[1] / (h / 10)), int(pt[0] / (w / 17))
            if 0 <= row < 10 and 0 <= col < 17:
                apple_array[row, col] = digit

    def find_roi_row_col(arr):
        result = []
        rows, cols = arr.shape

        # 가로의 합
        for i in range(rows):
            for j in range(cols):
                for k in range(j+1, cols+1):
                    sub_arr = arr[i, j:k]
                    if np.sum(sub_arr) == 10:
                        result.append(((i,j), (i,k-1)))

        # 세로의 합
        for i in range(cols):
            for j in range(rows):
                for k in range(j+1, rows+1):
                    sub_arr = arr[j:k, i]
                    if np.sum(sub_arr) == 10:
                        result.append(((j,i), (k-1,i)))

        return result   

    # 10이 되는 영역 찾기
    result = find_roi_row_col(apple_array)

    # 초록색 선 그리기
    used_positions = set()  # 이미 선이 그려진 숫자의 위치 저장
    if result:
        for start, end in result:
            # 이미 선이 그려진 숫자가 포함되면 건너뜀
            if start in used_positions or end in used_positions:
                continue

            # 시작 숫자의 중앙 좌표
            center_x1 = int(start[1] * (w / 17) + (w / 34) + x)
            center_y1 = int(start[0] * (h / 10) + (h / 20) + y)

            # 끝 숫자의 중앙 좌표
            center_x2 = int(end[1] * (w / 17) + (w / 34) + x)
            center_y2 = int(end[0] * (h / 10) + (h / 20) + y)

            # 선 두께 및 색상 설정
            thickness = 5  # 굵은 선
            color = (0, 255, 0)  # 초록색

            # 중앙을 기준으로 독립적인 선을 그림
            cv2.line(cropped, (center_x1 - x, center_y1 - y), (center_x2 - x, center_y2 - y), color, thickness)

            # 사용된 숫자 위치 저장
            used_positions.add(start)
            used_positions.add(end)

    # 화면에 결과 보여주기
    cv2.imshow("Result", cropped)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
