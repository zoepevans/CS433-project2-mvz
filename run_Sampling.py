import numpy as np
import pandas as pd

from helper_functions import load_data_standardized
from SamplingData import preprocess_sample_with_cur, preprocess_features_with_cur

X,y = load_data_standardized("3-body")
X_reduced = preprocess_features_with_cur(X, retain_ratio=1/3)
np.save("X_reduced_features.npy", X_reduced)

# recommended to not use this part as it take a lot of time (more than 1 day), if necessary uncomment it

# X_reduced_sample_features,selected_index = preprocess_sample_with_cur(X_reduced, retain_ratio=1/4)
# y_reduced_samples = y[selected_index]

# np.save("X_reduced_samples_features.npy", X_reduced_sample_features)
# np.save("y_reduced_samples.npy", y_reduced_samples)