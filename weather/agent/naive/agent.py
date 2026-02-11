"""
Naive baseline agent for Weather Temperature Prediction.

Strategy: repeat last known temperature for all 6 future hours.
"""
import numpy as np


class Agent:
    def __init__(self):
        pass

    def predict(self, X_test):
        """
        Predict temperatures for 10 cities over the next 6 hours.

        Args:
            X_test: numpy array of shape (10, 24, 8)
                - Axis 0: 10 cities
                - Axis 1: 24 hourly observations (oldest -> most recent)
                - Axis 2: 8 features (temperature is index 0)

        Returns:
            numpy array of shape (10, 6) - temperature predictions
        """
        # Repeat last known temperature for all 6 future hours
        return np.repeat(X_test[:, -1:, 0], 6, axis=1)  # (10, 6)
