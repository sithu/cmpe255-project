import numpy as np

from sklearn.linear_model import LinearRegression

class RegressionModel:
    def __init__(self) -> None:
        self.model = None
        self._fit()

    def _fit(self) -> None:
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        self.model = LinearRegression().fit(X, y)
        print('score: ', self.model.score(X, y))


    def predict(self, feature_a, feature_b):
        feature_a = int(feature_a)
        feature_b = int(feature_b)
        result = self.model.predict(np.array([[feature_a, feature_b]]))
        return result.tolist()
        
