Open CV Finger traking

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = handLms.landmark

            # Get index and thumb tip
            index_finger = landmarks[8]
            thumb = landmarks[4]

            ix, iy = int(index_finger.x * w), int(index_finger.y * h)
            tx, ty = int(thumb.x * w), int(thumb.y * h)

            # Draw circle on index
            cv2.circle(img, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

            # Move mouse
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)

            # Click if pinch detected
            distance = np.hypot(ix - tx, iy - ty)
            if distance < 30:
                pyautogui.click()
                cv2.putText(img, 'Click', (ix, iy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Finger Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

