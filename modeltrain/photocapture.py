import cv2
import numpy as np
import os
import time

save_folder = "D:\Data\Images\Train"
count = 0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 300)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif cv2.waitKey(1) & 0xFF == 32:
        count += 1
        cv2.imwrite(f'{save_folder}/Img{time.time()}.jpg', frame)
        print(count)

cap.release()
cv2.destroyAllWindows()
