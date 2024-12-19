import numpy as np
from helper_functions import load_data_standardized
from Kernel import plot_learning_curve_krr_with_all_errors

feature_choice = "2-body"  # Options: "2-body", "3-body", "4-body", "2+3-body", "2+3+4-body"
X,y = load_data_standardized(feature_choice)

plot_learning_curve_krr_with_all_errors(
    X, y,
    kernel="rbf", # Options: "polynomial", "rbf"
    kernel_params={"gamma": 16.0}, # for rbf : {"gamma": x}, for polynomial : {"degree": x, "coef0": y, "gamma": z}
    alpha=0.01,
    train_sizes=np.logspace(np.log10(0.0001), np.log10(0.005), 5),
    feature_choice= feature_choice
)