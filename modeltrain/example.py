from ultralytics import YOLO
import cv2

yolo = YOLO("../runs/detect/train5/weights/best.pt")

results = yolo.predict("test_image.jpg", imgsz=640, conf=0.4)

for result in results:
    img = result.plot()
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
