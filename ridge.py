import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def plot_learning_curve_with_test(X_train, X_test, y_train, y_test, title="Testing error for different lambdas for 3 bodies"):
    train_sizes = np.logspace(np.log10(0.1), np.log10(1.0), 10)
    train_errors = []
    test_errors = []
    lambda_values = np.logspace(-20, 20, 100)
    for lambda_ in lambda_values:

        model = Ridge(alpha=lambda_)
        model.fit(X_train, y_train)

        train_rmse = np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2))/ np.std(y_train)
        train_errors.append(train_rmse)

        test_rmse = np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2))/ np.std(y_train)
        test_errors.append(test_rmse)

    # Plot
    plt.figure()
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Lambda values (log scale)")
    plt.ylabel("Root Mean Squared Error / std")
    plt.grid()

    plt.plot(lambda_values, train_errors, color="r", label="Training error")
    plt.plot(lambda_values, test_errors, color="g", label="Test error")

    plt.legend(loc="best")
    plt.savefig("ridge.png")
    plt.show()