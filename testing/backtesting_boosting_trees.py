from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier

from projet.lib.data_functions.config import *

###############################
# COMPUTE MISCLASSIFICATION ERROR
###############################
results = {}
tree = AdaBoostClassifier(n_estimators=5000, learning_rate=1.0)
train_scores = []
test_scores = []
importances = []
alphas = np.logspace(-3, 2, 10)
for alpha in alphas:
    # alpha = 0.01
    tree.learning_rate = alpha
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))
    importances.append(tree.feature_importances_)


for k, importance in enumerate(importances):
    print "Feature importance for alpha=%s : %s " % (alphas[k], importance)
fig = plt.figure(1)
plt.title('Test scores as a function of lambda')
plt.xlabel('lambda')
plt.ylabel('Test scores')
plt.plot(alphas, test_scores)
plt.scatter(alphas, test_scores)
plt.xscale('log')
fig.show()

