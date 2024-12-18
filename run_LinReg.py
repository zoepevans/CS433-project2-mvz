from helper_functions import load_data_standardized,
from LinearReg import plot_learning_curve_kfold
from sklearn.linear_model import LinearRegression


feature_choice = "2-body"  # Options: "2-body", "3-body", "4-body", "2+3-body", "2+3+4-body"
X,y = load_data_standardized(feature_choice)
plot_learning_curve_kfold(LinearRegression(), X, y, title=f"Learning Curve ({feature_choice})")