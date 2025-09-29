import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self, dt=1/30.0, process_noise=0.01, measurement_noise=5.0, initial_uncertainty=1000.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.set_dt(dt)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        # Covariance and noise settings
        self.kf.P *= initial_uncertainty
        self.kf.Q *= process_noise
        self.kf.R *= measurement_noise

        # Initial state
        self.kf.x = np.array([0, 0, 0, 0])

    def set_dt(self, dt):
        self.dt = dt
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])


    def update(self, measurement):
        self.kf.update(np.array(measurement))

    def predict(self):
        self.kf.predict()
        return float(self.kf.x[0]), float(self.kf.x[1])
