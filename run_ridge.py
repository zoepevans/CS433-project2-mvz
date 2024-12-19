from helper_functions import load_data_standardized,
from ridge import plot_learning_curve_with_test
from sklearn.model_selection import train_test_split

feature_choice = "2-body"  # Options: "2-body", "3-body", "4-body", "2+3-body", "2+3+4-body"
X,y = load_data_standardized(feature_choice)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plot_learning_curve_with_test(X_train, X_test, y_train, y_test)