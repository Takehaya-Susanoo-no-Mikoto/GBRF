# Gradient Boosting on Random Forest
## Decision Tree, Random Forest, and Gradient Boosting Regression Forest
This code consists of three classes, DecisionTree, RandomForest, and GBRF. These classes implement decision tree, random forest, and gradient boosting regression forest algorithms in Python.

### DecisionTree class
DecisionTree class implements a decision tree regression algorithm. This class has two main functions: fit and predict. The fit function takes in training data (features and targets) and grows a decision tree based on the training data. The predict function takes in test data (features) and returns the predicted targets using the trained decision tree. This class also has several helper functions including entropy, most_common, best_split, information_gain, grow_tree, and traverse_tree.

### RandomForest class
RandomForest class implements a random forest regression algorithm. This class has two main functions: fit and predict. The fit function takes in training data (features and targets) and grows a random forest based on the training data. The predict function takes in test data (features) and returns the predicted targets using the trained random forest. This class also has several helper functions including fit, predict, information_gain, and traverse_tree.

### GBRF class
GBRF class implements a gradient boosting regression forest algorithm. This class has two main functions: fit and predict. The fit function takes in training data (features and targets) and grows a gradient boosting regression forest based on the training data. The predict function takes in test data (features) and returns the predicted targets using the trained gradient boosting regression forest. This class also has several helper functions including fit, predict, information_gain, and traverse_tree.

### Required libraries
This code requires the following libraries:

- scikit-learn for dataset generation
- numpy
- matplotlib

This algorithm can be used for regression tasks. It is based on the combination of random forests and gradient boosting, where each decision tree is built using a random subset of features and training examples, and the prediction is the weighted average of the trees' outputs. The algorithm is optimized using gradient descent on the mean squared error.

Algorithm can achieve good results if the hyperparameters are carefully tuned. Some of the important hyperparameters to consider include the number of estimators (i.e., the number of trees to build), the maximum depth of the trees, and the minimum number of samples required to split a node.

The algorithm is implemented using only NumPy. It is easy to use and can be adapted to various regression tasks.
![image](https://user-images.githubusercontent.com/124432421/236643232-16a4ac84-4fe0-47b5-8cb3-60dbe6ebc95d.png)
