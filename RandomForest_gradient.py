from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=2000)
X_train, X_test, y_train, y_test = train_test_split(X, y)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=20, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, features, targets):
        self.tree = self.grow_tree(features, targets)

    def predict(self, features):
        return np.array([self.travers_tree(feature, self.tree) for feature in features])

    def entropy(self, targets):
        c = np.sum(targets) / len(targets)
        MSE = np.sum((c - targets) ** 2) / len(targets)
        MAE = np.sum(np.abs(c - targets)) / len(targets)
        return MSE

    def most_common(self, targets):
        c = np.sum(targets) / len(targets)
        return c

    def best_split(self, features, targets):
        best_feature, best_threshold = None, None
        best_gain = -1

        index = np.random.choice(features.shape[1])

        for i in [index]:
            thresholds = np.unique(features[:, i])
            for threshold in thresholds:
                gain = self.information_gain(features[:, i], targets, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        return best_feature, best_threshold

    def information_gain(self, features_column, targets, threshold):
        n = len(targets)
        parent = self.entropy(targets)

        left_indexes = np.argwhere(features_column <= threshold).flatten()
        right_indexes = np.argwhere(features_column > threshold).flatten()

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return 0

        e_l, n_l = self.entropy(targets[left_indexes]), len(left_indexes)
        e_r, n_r = self.entropy(targets[right_indexes]), len(right_indexes)

        child = (n_l / n) * e_l + (n_r / n) * e_r
        return parent - child

    def grow_tree(self, features, targets, depth=0):
        n_samples = len(targets)
        n_labels = len(np.unique(targets))

        if n_labels == 1 or depth >= self.max_depth or n_samples <= self.min_samples:
            return Node(value=self.most_common(targets))

        best_feature, best_threshold = self.best_split(features, targets)

        left_indexes = np.argwhere(features[:, best_feature] <= best_threshold).flatten()
        right_indexes = np.argwhere(features[:, best_feature] > best_threshold).flatten()

        if len(left_indexes) == 0 or len(right_indexes) == 0:
            return Node(value=self.most_common(targets))

        left = self.grow_tree(features[left_indexes, :], targets[left_indexes], depth + 1)
        right = self.grow_tree(features[right_indexes, :], targets[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def travers_tree(self, feature, tree):
        if tree.is_leaf_node():
            return tree.value

        if feature[tree.feature] < tree.threshold:
            return self.travers_tree(feature, tree.left)
        return self.travers_tree(feature, tree.right)


class RandomForest:
    def __init__(self, n_trees=3, max_depth=100, min_samples=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.n_samples = None
        self.samples = []
        self.oob_samples = []
        self.oob_results = []

    def fit(self, features, targets):
        self.n_samples = features.shape[0]
        for i in range(self.n_trees):
            samples = np.random.choice(self.n_samples, self.n_samples, replace=True)
            oob_samples = np.array([j for j in range(self.n_samples) if j not in samples])

            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(features[samples, :], targets[samples])
            self.oob_results.append(tree.predict(features[oob_samples, :]))

            self.trees.append(tree)
            self.samples.append(samples)
            self.oob_samples.append(oob_samples)

    def predict(self, features):
        predictions = np.zeros(features.shape[0])
        for i in range(self.n_trees):
            predictions += self.trees[i].predict(features)
        return predictions / self.n_trees


class GBT:
    def __init__(self, n_estimators=100, max_depth=3, min_samples=10, lr=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.residuals = []
        self.lr = lr
        self.first_leaf = None
        self.train_score = []

    def fit(self, features, targets):
        self.first_leaf = targets.mean()
        predictions = np.ones(len(targets)) * self.first_leaf

        for i in range(self.n_estimators):
            residuals = targets - predictions
            self.residuals.append(residuals)

            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(features, residuals)
            self.trees.append(tree)

            predictions += self.lr * tree.predict(features)
            self.train_score.append(self.score(self.predict(features, len(self.trees)), targets))

    def predict(self, features, n):
        predictions = np.ones(features.shape[0]) * self.first_leaf

        for i in range(n):
            predictions += self.lr * self.trees[i].predict(features)

        return predictions

    def score(self, predicted, targets):
        return 1 - np.sum((predicted - targets) ** 2) / np.sum((targets.mean() - targets) ** 2)


reg = GBT(n_estimators=100)
reg.fit(X_train, y_train)

y_predicted = reg.predict(X_test, 100)

n = 5
y_test_sample = y_test[:n]
y_predicted_sample = y_predicted[:n]

print("True labels: ", y_test_sample)
print("Predicted labels: ", y_predicted_sample)
test_score = []

for i in range(reg.n_estimators):
    test_score.append(reg.score(reg.predict(X_test, i), y_test))

plt.plot(1 - np.array(reg.train_score), label="train error")
plt.plot(1 - np.array(test_score), label="test error")
plt.legend()
plt.ylabel("error")
plt.show()
