from projet.lib.data_functions.utils import *

clf = LogisticRegression()
clf.fit(X_train, y_train)
predicted_values = clf.predict(X_test)
predicted_probas = clf.predict_proba(X_test)
print (np.sum(predicted_values*y_test)+y_test.shape[0])*0.5/y_test.shape[0]

n = 0
total = 0
threshold = 0.7
for k in range(X_test.shape[0]):
    found = False
    for element in predicted_probas[k,:]:
        if element > threshold:
            found = True
            break
    if found:
        total += 1.0
        if y_test[k]*predicted_values[k] > 0:
            n += 1.0
print "proportion of good choice among those where proba was > %s : %s, total number proba>%s : %s, sample_size : %s" % (n/total, threshold, threshold, total, y_test.shape[0])

# regr = LinearRegression()
# regr.fit(X_train, np.squeeze(np.asarray(train_df.as_matrix(['next_roi_%s' % delta]))))
# good_trades = np.sign(regr.predict(X_test)*train_standard_factors['next_roi_%s'%delta][1] + train_standard_factors['next_roi_%s'%delta][0]) * y_test
# print (np.sum(good_trades) + good_trades.shape[0])*0.5/good_trades.shape[0]