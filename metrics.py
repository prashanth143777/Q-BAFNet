from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.stats import concordance_correlation_coefficient as ccc
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute standard regression metrics used in DTA tasks.

    Args:
        y_true (np.ndarray): Ground-truth values (1D).
        y_pred (np.ndarray): Predicted values (1D).

    Returns:
        dict: Dictionary with MSE, PCC, CI, and R2.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    ci = ccc(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": round(mse, 4),
        "PCC": round(pcc, 4),
        "CI": round(ci, 4),
        "R2": round(r2, 4)
    }

def print_metrics(metrics_dict, prefix=""):
    print(f"{prefix}MSE: {metrics_dict['MSE']:.4f}, "
          f"PCC: {metrics_dict['PCC']:.4f}, "
          f"CI: {metrics_dict['CI']:.4f}, "
          f"R²: {metrics_dict['R2']:.4f}")
