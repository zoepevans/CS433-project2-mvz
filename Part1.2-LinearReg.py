import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold

# Get the current directory of the notebook
current_dir = os.getcwd()

energies_path = os.path.join(current_dir, "data", "energies.npy")
features_2b_path = os.path.join(current_dir, "data", "features_2b.npy")
features_3b_path = os.path.join(current_dir, "data", "features_3b.npy")

energies = np.load(energies_path)
features_2bodies = np.load(features_2b_path)
features_3bodies = np.load(features_3b_path)
features_2and3bodies = np.hstack((features_2bodies, features_3bodies))

# Split the data
features_2b_train, features_2b_test, features_3b_train, features_3b_test, energies_train, energies_test = train_test_split(
    features_2bodies, features_3bodies, energies, test_size=0.2, random_state=42
)

print("Training set size:", features_2b_train.shape, features_3b_train.shape, energies_train.shape)
print("Test set size:", features_2b_test.shape, features_3b_test.shape, energies_test.shape)


# Define each feature combination in `X_train` and `X_test`
X_train_2 = features_2b_train
X_test_2 = features_2b_test

X_train_3 = features_3b_train
X_test_3 = features_3b_test

X_train_2_3 = np.hstack((features_2b_train, features_3b_train))
X_test_2_3 = np.hstack((features_2b_test, features_3b_test))

# Define a function to plot learning curves with a test set evaluation
def plot_learning_curve_with_test(model, X_train, X_test, y_train, y_test, title="Learning Curve"):
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_errors = []
    test_errors = []

    for train_size in train_sizes:
        
        train_subset_size = int(train_size * len(X_train))
        
        # Use only the subset of the training data
        X_train_subset = X_train[:train_subset_size]
        y_train_subset = y_train[:train_subset_size]
        
        model.fit(X_train_subset, y_train_subset)
        
        train_rmse = np.sqrt(np.mean((model.predict(X_train_subset) - y_train_subset) ** 2))
        train_errors.append(train_rmse)
        
        test_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))
        test_errors.append(test_rmse)

    # Plot the learning curves
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Root Mean Squared Error")
    plt.grid()

    plt.plot(train_sizes * len(X_train), train_errors, 'o-', color="r", label="Training error")
    plt.plot(train_sizes * len(X_train), test_errors, 'o-', color="g", label="Test error")
    
    plt.legend(loc="best")
    plt.show()

def plot_learning_curve_kfold(model, X, y, n_splits=5, title="Learning Curve with K-Fold CV"):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)

    all_train_errors = []
    all_test_errors = []

    # Plot each fold's learning curve
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        train_errors = []
        test_errors = []

        for train_size in train_sizes:
            train_subset_size = int(train_size * len(X_train))
            X_train_subset = X_train[:train_subset_size]
            y_train_subset = y_train[:train_subset_size]

            model.fit(X_train_subset, y_train_subset)
            train_rmse = np.sqrt(np.mean((model.predict(X_train_subset) - y_train_subset) ** 2))
            test_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))

            train_errors.append(train_rmse)
            test_errors.append(test_rmse)

        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)

        # Plot the fold's learning curve
        plt.plot(train_sizes * len(X_train), train_errors, 'o--', label=f"Fold {fold_idx + 1} Training", alpha=0.4, color="r")
        plt.plot(train_sizes * len(X_train), test_errors, 'o--', label=f"Fold {fold_idx + 1} Test", alpha=0.4, color="g")

    # Display each fold's learning curves together
    plt.title(f"{title} - Individual Folds")
    plt.xlabel("Training examples")
    plt.ylabel("Root Mean Squared Error")
    plt.grid()
    plt.legend(loc="best", ncol=2, fontsize="small")
    plt.show()

    # Calculate mean and standard deviation across folds
    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)

    train_errors_mean = all_train_errors.mean(axis=0)
    train_errors_std = all_train_errors.std(axis=0)
    test_errors_mean = all_test_errors.mean(axis=0)
    test_errors_std = all_test_errors.std(axis=0)

    # Plot the average learning curve with error bars
    plt.figure()
    plt.title(f"{title} - Mean with Error Bars")
    plt.xlabel("Training examples")
    plt.ylabel("Root Mean Squared Error")
    plt.grid()

    plt.fill_between(train_sizes * len(X_train), train_errors_mean - train_errors_std,
                     train_errors_mean + train_errors_std, alpha=0.1, color="r")
    plt.plot(train_sizes * len(X_train), train_errors_mean, 'o-', color="r", label="Average Training error")

    plt.fill_between(train_sizes * len(X_train), test_errors_mean - test_errors_std,
                     test_errors_mean + test_errors_std, alpha=0.1, color="g")
    plt.plot(train_sizes * len(X_train), test_errors_mean, 'o-', color="g", label="Average Test error")
    
    plt.legend(loc="best")
    plt.show()

plot_learning_curve_kfold(LinearRegression(), X_train_2_3, energies_train, title="Learning Curve (2+3-body features)")
# plot_learning_curve_kfold(LinearRegression(), X_train_3, energies_train, title="Learning Curve (3-body features)")
# plot_learning_curve_kfold(LinearRegression(), X_train_2, energies_train, title="Learning Curve (2-body features)")

# plot_learning_curve_with_test(LinearRegression(), X_train_2_3, X_test_2_3, energies_train, energies_test, title="Learning Curve (2+3-body features)")
# plot_learning_curve_with_test(LinearRegression(), X_train_3, X_test_3, energies_train, energies_test, title="Learning Curve (3-body features)")
# plot_learning_curve_with_test(LinearRegression(), X_train_2, X_test_2, energies_train, energies_test, title="Learning Curve (2-body features)")
