import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


class LinePredictor:
    def __init__(self, degree=1):
        self.degree = degree
        self.model = None
        self.mid_points = None

    def fit_polynomial(self, mid_points):
        """使用多项式回归拟合中线"""
        if len(mid_points) == 0:
            return None, None
        self.mid_points = np.array(mid_points)
        X = self.mid_points[:, 1].reshape(-1, 1)
        y = self.mid_points[:, 0]
        self.model = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
        self.model.fit(X, y)
        return self.model, self.mid_points

    def determine_track_type(self):
        """判断赛道类型"""
        if self.model is None:
            return "Unknown"
        coefficients = self.model.named_steps['linearregression'].coef_
        curvature = np.sum(np.abs(coefficients[1:]))
        if -0.28 < curvature < 0.28:
            return "Straight"
        elif np.mean(coefficients[1:]) < -0.28:
            return "Right Turn"
        elif np.mean(coefficients[1:]) > 0.28:
            return "Left Turn"
        else:
            return "Intersection"
