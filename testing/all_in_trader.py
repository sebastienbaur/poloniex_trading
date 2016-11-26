from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from projet.lib.data_functions.utils import *


def at_least_one_bigger(prediction, threshold):
    """
    prediction is a matrix ith one line and 2 or 3 columns
    That function returns True if one of the coefficient is bigger than threshold
    :param prediction:
    :param threshold:
    :return:
    """
    for element in prediction[0]:
        if element > threshold:
            return True
    else:
        return False

# train classifiers and regressors
clf1 = KNeighborsClassifier(n_neighbors=400)
clf1.fit(X_train, y_train)
clf2 = SVC(gamma=0.01, C=46.4158883336127772)
clf2.fit(X_train, y_train)
clf3 = LogisticRegression()
clf3.fit(X_train, y_train)
regr1 = LinearRegression()
regr1.fit(X_train, y_train)
clf4 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.01)
clf4.fit(X_train, y_train)

# initialize 3 traders
is_good_prediction_1 = []
is_good_prediction_2 = []
is_good_prediction_3 = []
is_good_prediction_4 = []
is_good_prediction_5 = []
predictions_1 = []
predictions_2 = []
predictions_3 = []
predictions_4 = []
predictions_5 = []
money1 = {'ETH': 400, 'BTC':10}
money2 = {'ETH': 400, 'BTC':10}
money3 = {'ETH': 400, 'BTC':10}
money4 = {'ETH': 400, 'BTC':10}
money5 = {'ETH': 400, 'BTC':10}
all_in_BTC_1 = [money1['BTC'] + money1['ETH'] * typical_prices_test[0]]  # this is a list that will contain the conversion of all money in bitcoin, at each trading step, for trader 1
all_in_BTC_2 = [money2['BTC'] + money2['ETH'] * typical_prices_test[0]]  # idem for trader 2
all_in_BTC_3 = [money3['BTC'] + money3['ETH'] * typical_prices_test[0]]  # idem for trader 3
all_in_BTC_4 = [money4['BTC'] + money4['ETH'] * typical_prices_test[0]]  # idem for trader 4
all_in_BTC_5 = [money5['BTC'] + money5['ETH'] * typical_prices_test[0]]  # idem for trader 4
ETH_1 = [money1['ETH']]
ETH_2 = [money2['ETH']]
ETH_3 = [money3['ETH']]
ETH_4 = [money4['ETH']]
ETH_5 = [money5['ETH']]
BTC_1 = [money1['BTC']]
BTC_2 = [money2['BTC']]
BTC_3 = [money3['BTC']]
BTC_4 = [money4['BTC']]
BTC_5 = [money5['BTC']]


# AND TRADE
for k in range(X_test.shape[0]):
    data = X_test[k, :]
    prediction1 = clf1.predict(data.reshape(1, -1))[0]
    prediction2 = clf2.predict(data.reshape(1, -1))[0]
    prediction3 = np.sign(regr1.predict(data.reshape(1, -1))[0] * scaling_factors['next_roi_%s' % delta][1] + scaling_factors['next_roi_%s' % delta][0])
    prediction4 = clf3.predict(data.reshape(1, -1))[0]
    prediction5 = clf4.predict(data.reshape(1, -1))[0]
    predictions_1.append(prediction1)
    predictions_2.append(prediction2)
    predictions_3.append(prediction3)
    predictions_4.append(prediction4)
    predictions_5.append(prediction5)

    if prediction1*y_test[k] > 0:
        is_good_prediction_1.append(1)
    else:
        is_good_prediction_1.append(0)
    if prediction2*y_test[k] > 0:
        is_good_prediction_2.append(1)
    else:
        is_good_prediction_2.append(0)
    if prediction3*y_test[k] > 0:
        is_good_prediction_3.append(1)
    else:
        is_good_prediction_3.append(0)
    if prediction4*y_test[k] > 0:
        is_good_prediction_4.append(1)
    else:
        is_good_prediction_4.append(0)
    if prediction5*y_test[k] > 0:
        is_good_prediction_5.append(1)
    else:
        is_good_prediction_5.append(0)

    # TRADER 1
    if prediction1 > 0:
        if money1['BTC'] > 0:
            money1['ETH'] += money1['BTC']/typical_prices_test[k]
            money1['BTC'] = 0
    else:
        if money1['ETH'] > 0:
            money1['BTC'] += money1['ETH']*typical_prices_test[k]
            money1['ETH'] = 0

    # TRADER 2
    if prediction2 > 0:
        if money2['BTC'] > 0:
            money2['ETH'] += money2['BTC']/typical_prices_test[k]
            money2['BTC'] = 0
    else:
        if money2['ETH'] > 0:
            money2['BTC'] += money2['ETH']*typical_prices_test[k]
            money2['ETH'] = 0

    # TRADER 3
    if prediction3 > 0:
        if money3['BTC'] > 0:
            money3['ETH'] += money3['BTC']/typical_prices_test[k]
            money3['BTC'] = 0
    else:
        if money3['ETH'] > 0:
            money3['BTC'] += money3['ETH']*typical_prices_test[k]
            money3['ETH'] = 0

    # TRADER 4
    if prediction4 > 0:
        if money4['BTC'] > 0:
            money4['ETH'] += money4['BTC']/typical_prices_test[k]
            money4['BTC'] = 0
    else:
        if money4['ETH']>0:
            money4['BTC'] += money4['ETH']*typical_prices_test[k]
            money4['ETH'] = 0

    # TRADER 5
    if prediction5 > 0:
        if money5['BTC'] > 0:
            money5['ETH'] += money5['BTC']/typical_prices_test[k]
            money5['BTC'] = 0
    else:
        if money5['ETH']>0:
            money5['BTC'] += money5['ETH']*typical_prices_test[k]
            money5['ETH'] = 0

    # update bitcoin capital of each trader
    all_in_BTC_1.append(money1['BTC'] + money1['ETH'] * typical_prices_test[k])
    all_in_BTC_2.append(money2['BTC'] + money2['ETH'] * typical_prices_test[k])
    all_in_BTC_3.append(money3['BTC'] + money3['ETH'] * typical_prices_test[k])
    all_in_BTC_4.append(money4['BTC'] + money4['ETH'] * typical_prices_test[k])
    all_in_BTC_5.append(money5['BTC'] + money5['ETH'] * typical_prices_test[k])
    ETH_1.append(money1['ETH'])
    ETH_2.append(money2['ETH'])
    ETH_3.append(money3['ETH'])
    ETH_4.append(money4['ETH'])
    ETH_5.append(money5['ETH'])
    BTC_1.append(money1['BTC'])
    BTC_2.append(money2['BTC'])
    BTC_3.append(money3['BTC'])
    BTC_4.append(money4['BTC'])
    BTC_5.append(money5['BTC'])

# PLOT RESULTS
f, axarr = plt.subplots(2, 2)
f.suptitle('Trader SVM (blue) vs Trader KNN (red) vs Linear regr (green) vs LR (magenta)')

axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_1)/all_in_BTC_1[0], c='r')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_2)/all_in_BTC_2[0], c='b')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_3)/all_in_BTC_3[0], c='g')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_4)/all_in_BTC_4[0], c='m')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_5)/all_in_BTC_5[0], c='k')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], [0 for k in range(len(all_in_BTC_1))], c='k')
axarr[0,0].set_title("Overall BTC profit")

axarr[0,1].hist(np.c_[is_good_prediction_1, is_good_prediction_2, is_good_prediction_3, is_good_prediction_4, is_good_prediction_5], color=['red', 'blue', 'green', 'magenta', 'black'])
axarr[0,1].set_title("Good choice vs bad choice")

axarr[1,0].hist(np.c_[predictions_1, predictions_2, predictions_3, predictions_4, predictions_5], color=['red', 'blue', 'green', 'magenta', 'black'])
axarr[1,0].set_title("Predictions")

axarr[1,1].plot([k for k in range(len(ETH_1))], -1 + np.array(ETH_1)/ETH_1[0], c='r')
axarr[1,1].plot([k for k in range(len(ETH_2))], -1 + np.array(ETH_2)/ETH_2[0], c='b')
axarr[1,1].plot([k for k in range(len(ETH_3))], -1 + np.array(ETH_3)/ETH_3[0], c='g')
axarr[1,1].plot([k for k in range(len(ETH_4))], -1 + np.array(ETH_4)/ETH_4[0], c='m')
axarr[1,1].plot([k for k in range(len(ETH_5))], -1 + np.array(ETH_5)/ETH_5[0], c='k')
axarr[1,1].set_title("ETH profit as a percentage of the initial value")

f.show()