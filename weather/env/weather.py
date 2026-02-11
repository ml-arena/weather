"""
Weather Temperature Prediction Environment - Core Logic.

Reusable environment class for testing and local development.
The env.py in environment_repos/weather/ delegates to this class.
"""
import numpy as np


class WeatherEnv:
    """Weather temperature prediction environment for supervised learning (predict-only)."""

    CITIES = [
        "Paris", "London", "New York", "Tokyo", "Sydney",
        "Dubai", "Mumbai", "Sao Paulo", "Moscow", "Beijing"
    ]
    FEATURES = [
        "temperature", "rain", "wind_speed", "wind_direction",
        "humidity", "clouds", "visibility", "snow"
    ]
    N_CITIES = 10
    N_HISTORY = 24
    N_FEATURES = 8
    N_PREDICT = 6

    def __init__(self, history, ground_truth):
        """
        Accept pre-processed numpy arrays.

        Args:
            history: numpy array of shape (10, 24, 8) - hourly weather features
            ground_truth: numpy array of shape (10, 6) - target temperatures
        """
        self.history = np.asarray(history, dtype=np.float32)
        self.ground_truth = np.asarray(ground_truth, dtype=np.float32)
        self.evaluated = False

    def reset(self):
        self.evaluated = False

    def get_next_task(self):
        if self.evaluated:
            return None
        return {
            'X_test': self.history,        # (10, 24, 8)
            'y_test': self.ground_truth    # (10, 6)
        }

    def evaluate(self, predictions, true_labels):
        self.evaluated = True
        predictions = np.array(predictions).reshape(self.N_CITIES, self.N_PREDICT)
        true_labels = np.array(true_labels).reshape(self.N_CITIES, self.N_PREDICT)
        mae = np.mean(np.abs(predictions - true_labels))
        return -mae  # Negated: higher = better

    def is_complete(self):
        return self.evaluated
