import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def plot_learning_curve_kfold(model, X, y, n_splits=5, title="Learning Curve with K-Fold CV", feature_choice="2-body",train_sizes = np.logspace(np.log10(0.01), np.log10(0.1), 3)):
    """
    Plot and save the learning curve for a given model using K-Fold Cross Validation.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_train_errors = []
    all_test_errors = []
    subset_sizes = []
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
            subset_sizes.append(train_subset_size)

        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)

        plt.plot(train_sizes * len(X_train), train_errors, 'o--', label=f"Fold {fold_idx + 1} Training", alpha=0.4, color="r")
        plt.plot(train_sizes * len(X_train), test_errors, 'o--', label=f"Fold {fold_idx + 1} Test", alpha=0.4, color="g")

    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"{title} - Individual Folds")
    plt.xlabel("Training examples")
    plt.ylabel("Root Mean Squared Error")
    plt.grid()
    plt.legend(loc="best", ncol=2, fontsize="small")
    plt.show()

    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)

    train_errors_mean = all_train_errors.mean(axis=0)
    train_errors_std = all_train_errors.std(axis=0)
    test_errors_mean = all_test_errors.mean(axis=0)
    test_errors_std = all_test_errors.std(axis=0)

    plt.figure()
    plt.title(f"{title} - Mean with Error Bars")
    plt.xscale("log")
    plt.yscale("log")
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

    np.save(f"train_errors_{feature_choice}_LinReg.npy", all_train_errors)
    np.save(f"test_errors_{feature_choice}_LinReg.npy", all_test_errors)
    np.save(f"subset_sizes_{feature_choice}_LinReg.npy", subset_sizes)
