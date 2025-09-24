import numpy as np
from sklearn.preprocessing import FunctionTransformer


def get_log1_scaler(**kwargs):
    def log_transform(x):
        return np.log1p(x)

    def inverse_log_transform(x_t):
        return np.expm1(x_t)

    return FunctionTransformer(func=log_transform, inverse_func=inverse_log_transform, **kwargs)
