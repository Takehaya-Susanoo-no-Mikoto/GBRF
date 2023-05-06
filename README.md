# Gradient Boosting on Random Forest
This algorithm can be used for regression tasks. It is based on the combination of random forests and gradient boosting, where each decision tree is built using a random subset of features and training examples, and the prediction is the weighted average of the trees' outputs. The algorithm is optimized using gradient descent on the mean squared error.

This algorithm can achieve good results if the hyperparameters are carefully tuned. Some of the important hyperparameters to consider include the number of estimators (i.e., the number of trees to build), the maximum depth of the trees, and the minimum number of samples required to split a node.

The algorithm is implemented using only NumPy. It is easy to use and can be adapted to various regression tasks.
