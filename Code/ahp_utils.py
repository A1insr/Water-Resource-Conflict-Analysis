# ahp_utils.py
import numpy as np


def ahp_weights(pairwise_matrix):
    """
    Compute AHP weights from a pairwise comparison matrix.

    Parameters
    ----------
    pairwise_matrix : list[list] or np.ndarray
        Pairwise comparison matrix provided by the user.

    Returns
    -------
    weight_vector : np.ndarray
        Vector of criterion weights (sum = 1).
    norm_matrix : np.ndarray
        Normalized matrix (columns sum to 1).
    """
    M = np.array(pairwise_matrix, dtype=float)

    # 1) Sum of each column
    col_sums = M.sum(axis=0)

    # 2) Normalize columns
    norm_matrix = M / col_sums

    # 3) Row averages give the weight vector
    weight_vector = norm_matrix.mean(axis=1)

    # Normalize to ensure sum = 1 (for safety)
    weight_vector = weight_vector / weight_vector.sum()

    return weight_vector, norm_matrix


def compute_utility(score_dict, weight_dict):
    """
    Compute utility using named criteria dictionaries.

    Parameters
    ----------
    score_dict : dict
        Example: {"agricultural_income": 80, "social_satisfaction": 90}
    weight_dict : dict
        Example: {"agricultural_income": 0.4, "social_satisfaction": 0.6}

    Returns
    -------
    float
        Weighted utility value.
    """
    total = 0.0
    for criterion, score in score_dict.items():
        total += score * weight_dict[criterion]
    return total


def normalize_to_minus5_plus5(values):
    """
    Normalize utility array to [-5, +5].

    Parameters
    ----------
    values : list or np.ndarray

    Returns
    -------
    np.ndarray
        Scaled values in the range [-5, +5].
    """
    values = np.array(values, dtype=float)
    v_min = values.min()
    v_max = values.max()

    if np.isclose(v_min, v_max):
        return np.zeros_like(values)

    return -5 + 10 * (values - v_min) / (v_max - v_min)
