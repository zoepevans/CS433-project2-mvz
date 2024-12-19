from skmatter.sample_selection import CUR

def preprocess_sample_with_cur(X, retain_ratio=1 / 3):
    """
    Reduce the number of rows using CUR sampling.
    Parameters:
    - X: Input data array of shape (n_samples, n_features).
    - retain_ratio: Fraction of samples to retain (default is 1/3).
    Returns:
    - Reduced sample array.
    """
    n_samples_to_select = int(X.shape[0] * retain_ratio)
    sampler = CUR(n_to_select=n_samples_to_select)
    sampler.fit(X)
    selected_indices = sampler.get_support(indices=True)
    return X[selected_indices], selected_indices


def preprocess_features_with_cur(X, retain_ratio=1 / 3):
    """
    Reduce the number of features using CUR sampling.
    Parameters:
    - X: Input data array of shape (n_samples, n_features).
    - retain_ratio: Fraction of features to retain (default is 1/3).
    Returns:
    - Reduced feature array with selected features.
    """
    n_features_to_select = int(X.shape[1] * retain_ratio)
    sampler = CUR(n_to_select=n_features_to_select)
    sampler.fit(X.T)
    selected_features = sampler.get_support(indices=True)
    return X[:, selected_features]