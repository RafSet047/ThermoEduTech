from typing import Iterable, Dict 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def regression_report(y_true: Iterable, y_pred: Iterable) -> Dict:
    """
    Generates a report of key regression metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    Returns:
        results (dict): A dictionary containing various regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    report = {
        'Mean Absolute Error (MAE)': float(mae),
        'Mean Squared Error (MSE)': float(mse),
        'Root Mean Squared Error (RMSE)': float(rmse),
        "Mean Absolute Percentage Error (MAPE)": float(mape),
        'R2 Score': float(r2)
    }
    return report