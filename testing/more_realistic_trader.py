from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from projet.lib.data_functions.utils import *

market = PoloniexMarket()
pair = "BTC_ETH"
# READ
data = pd.read_csv("trades_data_2.csv", sep=';', index_col=0, header=0)
read = True
# OR CREATE...
# data = market.get_trade_history(pair, 1435699200)
# ... AND SAVE
# data.to_csv('trades_data_2.csv', sep=';')
dates, typical_prices, volumes = typical_prices_from_trades_history(data, read=read)
predictors = predictors_from_typical_prices(typical_prices, volumes, scaling_factors)
predictors = parse_predictor_dataframe(predictors)
predictors = predictors.dropna()
X_test = predictors.as_matrix(X_columns)
y_test = np.squeeze(np.asarray(predictors.as_matrix(['next_roi_%s' % delta])))
y_test = np.sign(y_test * scaling_factors['next_roi_%s' % delta][1] + scaling_factors['next_roi_%s' % delta][0])


# train classifiers and regressors
knn = KNeighborsClassifier(n_neighbors=100, weights='distance')
knn.fit(X_train, y_train)
svm = SVC(gamma=0.01, C=46.4158883336127772)
svm.fit(X_train, y_train)
logistic_regr = LogisticRegression()
logistic_regr.fit(X_train, y_train)
linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train)



# initialize 3 traders
money1 = {'ETH': 400, 'BTC':10}
money2 = {'ETH': 400, 'BTC':10}
money3 = {'ETH': 400, 'BTC':10}
money4 = {'ETH': 400, 'BTC':10}
is_good_prediction_1 = []
is_good_prediction_2 = []
is_good_prediction_3 = []
is_good_prediction_4 = []
all_in_BTC_1 = [money1['BTC'] + money1['ETH'] * typical_prices[0]]  # this is a list that will contain the conversion of all money in bitcoin, at each trading step, for trader 1
all_in_BTC_2 = [money2['BTC'] + money2['ETH'] * typical_prices[0]]  # idem for trader 2
all_in_BTC_3 = [money3['BTC'] + money3['ETH'] * typical_prices[0]]  # idem for trader 3
all_in_BTC_4 = [money4['BTC'] + money4['ETH'] * typical_prices[0]]  # idem for trader 4
ETH_1 = [money1['ETH']]
ETH_2 = [money2['ETH']]
ETH_3 = [money3['ETH']]
ETH_4 = [money4['ETH']]
BTC_1 = [money1['BTC']]
BTC_2 = [money2['BTC']]
BTC_3 = [money3['BTC']]
BTC_4 = [money4['BTC']]

# AND TRADE
for k in range(X_test.shape[0]):
    data = X_test[k, :]
    prediction1 = knn.predict(data.reshape(1, -1))[0]
    prediction2 = svm.predict(data.reshape(1, -1))[0]
    prediction3 = np.sign(linear_regr.predict(data.reshape(1, -1))[0] * scaling_factors['next_roi_%s' % delta][1] + scaling_factors['next_roi_%s' % delta][0])
    prediction4 = logistic_regr.predict(data.reshape(1, -1))[0]
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

    # TRADER 1
    if prediction1 > 0:
        if money1['BTC'] > 0:
            money1['ETH'] += money1['BTC']/typical_prices[k]
            money1['BTC'] = 0
    else:
        if money1['ETH'] > 0:
            money1['BTC'] += money1['ETH']*typical_prices[k]
            money1['ETH'] = 0

    # TRADER 2
    if prediction2 > 0:
        if money2['BTC'] > 0:
            money2['ETH'] += money2['BTC']/typical_prices[k]
            money2['BTC'] = 0
    else:
        if money2['ETH'] > 0:
            money2['BTC'] += money2['ETH']*typical_prices[k]
            money2['ETH'] = 0

    # TRADER 3
    if prediction3 > 0:
        if money3['BTC'] > 0:
            money3['ETH'] += money3['BTC']/typical_prices[k]
            money3['BTC'] = 0
    else:
        if money3['ETH'] > 0:
            money3['BTC'] += money3['ETH']*typical_prices[k]
            money3['ETH'] = 0

    # TRADER 4
    if prediction4 > 0:
        if money4['BTC'] > 0:
            money4['ETH'] += money4['BTC']/typical_prices[k]
            money4['BTC'] = 0
    else:
        if money4['ETH']>0:
            money4['BTC'] += money4['ETH']*typical_prices[k]
            money4['ETH'] = 0

    # update bitcoin capital of each trader
    all_in_BTC_1.append(money1['BTC'] + money1['ETH'] * typical_prices[k])
    all_in_BTC_2.append(money2['BTC'] + money2['ETH'] * typical_prices[k])
    all_in_BTC_3.append(money3['BTC'] + money3['ETH'] * typical_prices[k])
    all_in_BTC_4.append(money4['BTC'] + money4['ETH'] * typical_prices[k])
    ETH_1.append(money1['ETH'])
    ETH_2.append(money2['ETH'])
    ETH_3.append(money3['ETH'])
    ETH_4.append(money4['ETH'])
    BTC_1.append(money1['BTC'])
    BTC_2.append(money2['BTC'])
    BTC_3.append(money3['BTC'])
    BTC_4.append(money4['BTC'])

# PLOT RESULTS
f, axarr = plt.subplots(2, 2)
f.suptitle('Trader SVM (blue) vs Trader KNN (red) vs Linear regr (green) vs LR (magenta)')

axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_1)/all_in_BTC_1[0], c='r')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_2)/all_in_BTC_2[0], c='b')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_3)/all_in_BTC_3[0], c='g')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_4)/all_in_BTC_4[0], c='m')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], [0 for k in range((len(all_in_BTC_1)))], c='k')
axarr[0,0].set_title("Overall BTC profit")

axarr[0,1].hist(np.c_[is_good_prediction_1, is_good_prediction_2, is_good_prediction_3, is_good_prediction_4], color=['red', 'blue', 'green', 'magenta'])
axarr[0,1].set_title("Good choice vs bad choice")

axarr[1,0].plot([k for k in range(len(BTC_1))], -1 + np.array(BTC_1)/BTC_1[0], c='r')
axarr[1,0].plot([k for k in range(len(BTC_2))], -1 + np.array(BTC_2)/BTC_2[0], c='b')
axarr[1,0].plot([k for k in range(len(BTC_3))], -1 + np.array(BTC_3)/BTC_3[0], c='g')
axarr[1,0].plot([k for k in range(len(BTC_4))], -1 + np.array(BTC_4)/BTC_4[0], c='m')
axarr[1,0].set_title("BTC profit as a percentage of the initial value")

axarr[1,1].plot([k for k in range(len(ETH_1))], -1 + np.array(ETH_1)/ETH_1[0], c='r')
axarr[1,1].plot([k for k in range(len(ETH_2))], -1 + np.array(ETH_2)/ETH_2[0], c='b')
axarr[1,1].plot([k for k in range(len(ETH_3))], -1 + np.array(ETH_3)/ETH_3[0], c='g')
axarr[1,1].plot([k for k in range(len(ETH_4))], -1 + np.array(ETH_4)/ETH_4[0], c='m')
axarr[1,1].set_title("ETH profit as a percentage of the initial value")

f.show()