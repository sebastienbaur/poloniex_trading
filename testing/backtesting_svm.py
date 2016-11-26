from itertools import product

from sklearn.svm import SVC

from projet.lib.data_functions.config import *

###############################
# COMPUTE MISCLASSIFICATION ERROR
###############################
compteur = 0
results = {}
for gamma, c in product(np.logspace(-3, 1, num=40, base=10), np.logspace(-1, 2, num=10)):
    results[(gamma, c)] = {}
    clf = SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(X_train, y_train)
    predicted_returns = clf.predict(X_test)
    is_good_trade = predicted_returns*y_test
    results[(gamma, c)]['%ssuccess' % '%'] = (np.sum(is_good_trade)+len(is_good_trade))*0.5/len(is_good_trade)
    print {'gamma': gamma, 'c': c}, results[(gamma, c)]



