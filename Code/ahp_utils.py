# ahp_utils.py
import numpy as np

RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

def ahp_weights(pairwise_matrix):
    """
    Column-normalization + row-mean (common AHP approximation).
    Returns (weights, normalized_matrix).
    """
    M = np.array(pairwise_matrix, dtype=float)
    col_sums = M.sum(axis=0)
    norm_matrix = M / col_sums
    weights = norm_matrix.mean(axis=1)
    weights = weights / weights.sum()
    return weights, norm_matrix

def ahp_weights_eigen(pairwise_matrix):
    """
    Principal eigenvector method (Saaty).
    Returns (weights, lambda_max).
    """
    M = np.array(pairwise_matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    idx = np.argmax(eigenvalues.real)
    lambda_max = float(eigenvalues.real[idx])
    w = np.abs(eigenvectors[:, idx].real)
    w = w / w.sum()
    return w, lambda_max

def ahp_consistency(pairwise_matrix, method="eigen"):
    """
    Compute weights and consistency metrics (lambda_max, CI, CR).
    method:
      - "eigen" (default): weights from principal eigenvector
      - "mean": weights from row-mean method (lambda_max via eigenvalues)
    """
    M = np.array(pairwise_matrix, dtype=float)
    n = M.shape[0]

    if method == "mean":
        weights, _ = ahp_weights(M)
        lambda_max = float(np.max(np.linalg.eigvals(M).real))
    else:
        weights, lambda_max = ahp_weights_eigen(M)

    if n <= 2:
        return weights, lambda_max, 0.0, 0.0

    CI = (lambda_max - n) / (n - 1)
    RI = RI_TABLE.get(n, 0.0)
    CR = 0.0 if RI == 0.0 else CI / RI
    return weights, lambda_max, float(CI), float(CR)

def compute_utility(score_dict, weight_dict):
    return float(sum(score_dict[c] * weight_dict[c] for c in weight_dict))

def normalize_to_minus5_plus5(values):
    values = np.array(values, dtype=float)
    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_min, v_max):
        return np.zeros_like(values)
    return -5 + 10 * (values - v_min) / (v_max - v_min)
