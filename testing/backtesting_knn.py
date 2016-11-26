from sklearn.neighbors import KNeighborsClassifier

from projet.lib.data_functions.config import *

###############################
# COMPUTE MISCLASSIFICATION ERROR
###############################
results = {}
knn = KNeighborsClassifier(weights='distance')
for k in range(1, 101):
    knn.n_neighbors = k
    knn.fit(X_train, y_train)
    predicted_returns = knn.predict(X_test)
    results[k] = knn.score(X_test, y_test)
    print "For k = %s, %ssuccess = %s" % (k, '%', results[k])

for k in range(0, 41):
    knn.n_neighbors = 10*k + 100
    knn.fit(X_train, y_train)
    predicted_returns = knn.predict(X_test)
    results[10*k + 100] = knn.score(X_test, y_test)
    print "For k = %s, %ssuccess = %s" % (10*k+100, '%', results[10*k+100])

