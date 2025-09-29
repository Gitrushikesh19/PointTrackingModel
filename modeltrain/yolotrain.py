from ultralytics import YOLO

if __name__ == '__main__':

    yolo = YOLO("../yolov8n.pt")
    model = yolo.train(data='config.yaml', epochs=50)
