import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        check_subsample_size: bool = True,
        random_state: int = None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.check_subsample_size = check_subsample_size
        self.random_state = random_state

        self.estimators = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.estimators = []

        if (
                self.check_subsample_size and
                (self.feature_subsample_size is not None) and
                not (0 <= self.feature_subsample_size <= X.shape[1])
        ):
            raise ValueError('Incorrect feature_subsample_size')

        for _ in range(self.n_estimators):
            idx = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)

            tree = DecisionTreeRegressor(
                criterion='squared_error',
                splitter='random',
                max_depth=self.max_depth,
                max_features=self.feature_subsample_size,
                random_state=self.random_state,
            )
            tree.fit(X[idx], y[idx])
            self.estimators.append(tree)

        return self

    def predict(self, X, estimators_c: int = None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        estimators_c = self.n_estimators if estimators_c is None else estimators_c

        pred = np.vstack([tree.predict(X) for _, tree in zip(range(estimators_c), self.estimators)])
        return np.mean(pred, axis=0)


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            check_subsample_size: bool = True,
            random_state: int = None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.check_subsample_size = check_subsample_size
        self.random_state = random_state

        self.estimators = []
        self.alphas = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        self.estimators = []

        if (
                self.check_subsample_size and
                (self.feature_subsample_size is not None) and
                not (0 <= self.feature_subsample_size <= X.shape[1])
        ):
            raise ValueError('Incorrect feature_subsample_size')

        S = y / self.learning_rate
        f = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                splitter='best',
                max_features=self.feature_subsample_size,
                random_state=self.random_state,
                max_depth=self.max_depth
            )
            tree.fit(X, S)

            pred = tree.predict(X)

            alpha = minimize_scalar(
                lambda a: np.sum((y - (f + a * pred)) ** 2)
            ).x

            f += alpha * self.learning_rate * pred

            S = y - f

            self.estimators.append(tree)
            self.alphas.append(alpha)

            print(f"{i=:3}\t{np.linalg.norm(S)=:.12f}")

    def predict(self, X, estimators_c: int = None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        estimators_c = len(self.estimators) if estimators_c is None else estimators_c
        est_gen = zip(range(estimators_c), self.estimators, self.alphas)

        pred = np.vstack([alpha * self.learning_rate * tree.predict(X) for _, tree, alpha in est_gen])
        return np.sum(pred, axis=0)
