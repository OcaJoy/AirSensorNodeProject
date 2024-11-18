import numpy as np

#####################################################################################
### KALMAN FILTER ####
######################
class KalmanFilter():
    def __init__(self, sensor_dim, Q, R, initial_reading=None):
        '''
        Initialize the Kalman Filter
        
        Paramters:
        state_dim (int): Dimension of the state vector.
        measurement_dim (int): Dimension of the measurement vector.
        Q (numpy.ndarray): Process noise covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        '''
        self.sensor_dim = sensor_dim
        
        # State Transition Matrix (A) 
        self.A = np.eye(sensor_dim) # Identity matrix since reading directly reflect sensor values over time
        
        # Measurement Matrix (C) 
        self.C = np.eye(sensor_dim) # Identity matrix since there is no transformation between state and measurment
        
        # Process Noise Covariance:
        self.Q = np.eye(sensor_dim)*Q
        
        # Measurement Noise Covariance Matrix (R)
        self.R = R
        
        # Initial State Estimate (x)
        if initial_reading is None:
            self.x = np.zeros(sensor_dim)
        else:
            self.x = initial_reading
        
        # Initial State Covariance (P)
        self.P = np.eye(sensor_dim)
        
    def predict(self):
        '''
        Predict next state and uncertainty
        '''
        # State prediction
        self.x = self.A @ self.x
        
        # Update estimate ucnertainty
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, z):
        """
        Update the state based on the measurement.
        
        Parameters:
            z (numpy.ndarray): Measurement vector.
        """
        # Kalman Gain:
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
        
        # Update step
        self.x = self.x + K @ (z-self.C @ self.x) # Correct the prediction
        self.P = (np.eye(self.sensor_dim) - K @ self.C) @ self.P # Update Covariance
    
    def filter(self, z):
        """
        Run full Kalman Filter Cycle with prediction and update.
        """
        self.predict()  # Prediction step
        self.update(z)  # Update step
        return self.x
