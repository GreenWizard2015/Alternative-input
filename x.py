import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import pywintypes # should be imported before win32api due to a bug in pywin32
import pywinauto
from pywinauto import win32functions, win32defines
from scipy.spatial.distance import cdist


# Инициализация MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

screen = np.array([
  win32functions.GetSystemMetrics(win32defines.SM_CXSCREEN),
  win32functions.GetSystemMetrics(win32defines.SM_CYSCREEN)
])
# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Обработка изображения и поиск рук
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Получение координаты указательного пальца
            finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            xy = np.array([1. - finger.x, finger.y])
            xy = np.multiply(xy, screen)
            x, y = tuple(xy.astype(np.int32))

            # if fist is closed then hold left mouse button
            fingerB = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            fingerC = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            fingerD = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            fingerE = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            # find avg distance between fingers using euclidean distance cdist
            fingers = np.array([
              [finger.x, finger.y],
              [fingerB.x, fingerB.y],
              [fingerC.x, fingerC.y],
              [fingerD.x, fingerD.y],
              [fingerE.x, fingerE.y]
            ])

            # dist = cdist(fingers, fingers, 'euclidean')
            # if dist.mean() < 0.05:
            #     pywinauto.mouse.press(button='left')
            
            pywinauto.mouse.move(coords=(x, y))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Показать изображение
    cv2.imshow('Hand Tracking', frame)
    # break on escape
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
