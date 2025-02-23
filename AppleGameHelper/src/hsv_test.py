import cv2
import numpy as np
import pyautogui

# 화면 캡처
screenshot = np.array(pyautogui.screenshot())
screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

# HSV 변환
hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)

# 마우스 클릭 이벤트 함수
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        print(f"Clicked HSV: {pixel}")  # 클릭한 픽셀의 HSV 값 출력

# 이미지 출력
cv2.imshow("Select HSV", screenshot)
cv2.setMouseCallback("Select HSV", pick_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
