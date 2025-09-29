import cv2
import time
from ultralytics import YOLO
from detector.kalman_tracker import KalmanTracker
from threaded_camera_reader import ThreadedCamera

def main(camera_src=0,
         Img_size=640,
         YOLO_INTERVAL=2,
         conf_thres=0.4,
         device='cuda',
         target_labels=("Bottle")):

    model = YOLO('../runs/detect/train5/weights/best.pt')

    try:
        model.to(device=device)
    except Exception as e:
        print("Couldn't move model to device:", e)

    cam = ThreadedCamera(src=camera_src)
    time.sleep(0.2)

    fps_est = 20.0
    dt_init = 1.0/fps_est
    tracker = KalmanTracker(dt=dt_init)

    frame_count = 0
    last_yolo_time = time.time()
    last_frame_time = time.time()
    detected_center = None
    pred_x, pred_y = 0, 0

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        display_frame = frame.copy()

        current = time.time()
        dt = current - last_frame_time if last_frame_time else dt_init
        last_frame_time = current

        if dt <= 0 or dt > 0.5:
            dt = dt_init
        tracker.set_dt(dt)

        frame_count += 1

        if frame_count % YOLO_INTERVAL == 0:
            t0 = time.time()
            results = model.predict(source=frame, imgsz=(Img_size, Img_size), conf=conf_thres, verbose=False)
            # if results:
            #     print('yes')
            t1 = time.time()
            last_yolo_time = t1

            if len(results) > 0:
                result = results[0]

                try:
                    boxes_data = result.boxes.data.cpu().numpy()
                except Exception:
                    boxes_data = result.boxes.data.numpy()

                chosen = None
                chosen_score = -1.0

                for row in boxes_data:
                    x1, y1, x2, y2, score, cls = row
                    cls = int(cls)
                    label = result.names.get(cls, str(cls)) if hasattr(result, 'names') else model.names.get(cls, str(cls))

                    if label in target_labels:
                        if score > chosen_score:
                            chosen_score = float(score)
                            chosen = (x1, y1, x2, y2, score, cls)

                if chosen is not None:
                    x1, y1, x2, y2, score, cls = chosen
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    detected_center = (cx, cy)

                    tracker.update(detected_center)

                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                else:
                    detected_center = None
            else:
                detected_center = None

        pred_x, pred_y = tracker.predict()

        cv2.circle(display_frame, (int(pred_x), int(pred_y)), 8, (0, 0, 255), -1)

        cv2.imshow("Frame", display_frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

