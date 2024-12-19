import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from skmatter.sample_selection import CUR


def plot_learning_curve_krr_with_all_errors(
    X, y, kernel="rbf", alpha=1.0, kernel_params=None, n_splits=5, train_sizes=np.logspace(np.log10(0.01), np.log10(1), 8),feature_choice="2-body"
):
    """
    Plot and save the learning curve for Kernel Ridge Regression and another plot with all the fold.
    """

    if kernel_params is None:
        kernel_params = {}
    #set up the Kfold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    #initialize the arrays to store the errors
    all_train_errors = []
    all_test_errors = []
    subset_sizes = []

    #loop through the train sizes
    for train_size in train_sizes:
        #initialize the arrays to store the errors for each fold
        fold_train_errors = []
        fold_test_errors = []

        #loop through the folds
        for train_idx, test_idx in kf.split(X):
            #split the data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            #select the subset of the data and store it
            subset_size = int(train_size * len(X_train))
            if subset_size not in subset_sizes:
                subset_sizes.append(subset_size)

            #sample the data to apply the kernel ridge regression
            sampler = CUR(n_to_select=subset_size)
            sampler.fit(X_train)
            selected_indices = sampler.get_support(indices=True)

            #fit the model
            X_train_subset = X_train[selected_indices]
            y_train_subset = y_train[selected_indices]
            model = KernelRidge(kernel=kernel, alpha=alpha, **kernel_params)
            model.fit(X_train_subset, y_train_subset)

            y_train_pred = model.predict(X_train_subset)
            y_test_pred = model.predict(X_test)

            fold_train_errors.append(root_mean_squared_error(y_train_subset, y_train_pred)/np.std(y_train))
            fold_test_errors.append(root_mean_squared_error(y_test, y_test_pred)/np.std(y_test))

        all_train_errors.append(fold_train_errors)
        all_test_errors.append(fold_test_errors)

    all_train_errors = np.array(all_train_errors)
    all_test_errors = np.array(all_test_errors)

    np.save(f"train_errors_krr_CUR_{feature_choice}_{kernel_params}.npy", all_train_errors.T)
    np.save(f"test_errors_krr_CUR_{feature_choice}_{kernel_params}.npy", all_test_errors.T)
    np.save(f"subset_sizes_krr_CUR_{feature_choice}_{kernel_params}.npy", subset_sizes)

    train_means = np.mean(all_train_errors, axis=1)
    test_means = np.mean(all_test_errors, axis=1)
    train_stds = np.std(all_train_errors, axis=1)
    test_stds = np.std(all_test_errors, axis=1)

    subset_sizes = np.array(subset_sizes[:len(train_means)])

    # Plotting learning curve with the standard deviation   
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        subset_sizes, train_means, yerr=train_stds, fmt='o-', label="Training Error", color="r", capsize=3
    )
    plt.errorbar(
        subset_sizes, test_means, yerr=test_stds, fmt='o-', label="Testing Error", color="g", capsize=3
    )
    plt.fill_between(
        subset_sizes,
        train_means - train_stds,
        train_means + train_stds,
        color="r", alpha=0.2, label="Train Error Std Dev"
    )
    plt.fill_between(
        subset_sizes,
        test_means - test_stds,
        test_means + test_stds,
        color="g", alpha=0.2, label="Test Error Std Dev"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Subset Size (Number of Training Samples)")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve for Kernel Ridge Regression with CUR Sampling")
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting all folds
    plt.figure(figsize=(10, 6))
    for idx, fold_train_errors in enumerate(all_train_errors.T):  # Transpose for fold-wise errors
        plt.plot(subset_sizes, fold_train_errors, label=f"Fold {idx + 1} - Train", color="r", alpha=0.6)
    for idx, fold_test_errors in enumerate(all_test_errors.T):  # Transpose for fold-wise errors
        plt.plot(subset_sizes, fold_test_errors, label=f"Fold {idx + 1} - Test", color="g", alpha=0.6)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Subset Size (Number of Training Samples)")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve for Kernel Ridge Regression (Subset Size) - All Folds")
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1))
    plt.grid()
    plt.show()
