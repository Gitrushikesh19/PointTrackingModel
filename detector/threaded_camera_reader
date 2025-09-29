import threading
import cv2
import time

class ThreadedCamera:
    def __init__(self, src=0, width=None, height=None):
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(3, width)
        if height:
            self.cap.set(4, height)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=0.5)
        self.cap.release()
