from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from projet.lib.data_functions.utils import *


def at_least_one_bigger(prediction, threshold):
    for element in prediction[0]:
        if element > threshold:
            return True
    else:
        return False


###############################
# GET DATA
# #############################
market = PoloniexMarket()
start_train_date = '2016/03/01'
end_train_date = '2016/05/01'
start_test_date = end_train_date
end_test_date = '2016/05/30'
pair = "BTC_ETH"
## LOAD ...
test_market_data = pd.read_csv('test_data 2016-05-01 2016-05-30.csv', sep=';', index_col=0, header=0)
train_market_data = pd.read_csv('train_data 2016-03-01 2016-05-01.csv', sep=';', index_col=0, header=0)

## ... OR CREATE AND ...
# train_market_data = market.get_historical_data(pair, datestring_to_timestamp(start_train_date), end=datestring_to_timestamp(end_train_date))
# test_market_data = market.get_historical_data(pair, datestring_to_timestamp(start_test_date), end=datestring_to_timestamp(end_test_date))
## ... SAVE
# train_market_data.to_csv('train_data 2016-03-01 2016-05-01.csv', sep=';')
# test_market_data.to_csv('test_data 2016-05-01 2016-05-30.csv', sep=';')

train_df, train_standard_factors = train_predictors(train_market_data, ALL_TYPICAL_FUNCTIONS, ALL_HLC_FUNCTIONS, ALL_TYPICAL_NAMES, ALL_HLC_NAMES)
train_df = parse_predictor_dataframe(train_df)
train_df = train_df.dropna()
test_df, test_standard_factors = train_predictors(test_market_data, ALL_TYPICAL_FUNCTIONS, ALL_HLC_FUNCTIONS, ALL_TYPICAL_NAMES, ALL_HLC_NAMES)
test_df = parse_predictor_dataframe(test_df)
test_df = test_df.dropna()


X_columns = [
    'prev_roi_1',
    'prev_roi_2',
    'prev_roi_3',
    'prev_roi_4',
    'prev_roi_5',
    'exp_avg_inf',
    'exp_avg_%s' % SMALL_PERIOD,
    'exp_avg_%s' % BIG_PERIOD,
    'mov_avg_%s' % SMALL_PERIOD,
    'mov_avg_%s' % BIG_PERIOD,
    'vol_%s' % SMALL_PERIOD,
    'vol_%s' % BIG_PERIOD,
    'moment_3_%s' % SMALL_PERIOD,
    'moment_3_%s' % BIG_PERIOD,
    'moment_4_%s' % SMALL_PERIOD,
    'moment_4_%s' % BIG_PERIOD,
    'trix',
    'rsi',
    'percent_r_%s' % SMALL_PERIOD,
    'percent_r_%s' % BIG_PERIOD,
    'ult_osc',
    'cci_%s' % SMALL_PERIOD,
    'cci_%s' % BIG_PERIOD,
    "volume",
    'macd'
]

delta = 1
X_train = train_df.as_matrix(X_columns)
y_train = np.squeeze(np.asarray(train_df.as_matrix(['next_roi_%s' % delta])))  # this is scaled
y_train = np.sign(y_train * train_standard_factors['next_roi_%s' % delta][1] + train_standard_factors['next_roi_%s' % delta][0])
X_test = test_df.as_matrix(X_columns)
y_test = np.squeeze(np.asarray(test_df.as_matrix(['next_roi_%s' % delta])))
y_test = np.sign(y_test * test_standard_factors['next_roi_%s' % delta][1] + test_standard_factors['next_roi_%s' % delta][0])
# don't forget to unscale the variables
typical_prices_test = np.squeeze(np.asarray(test_df.as_matrix(['typical_price']))) * test_standard_factors['typical_price'][1] + test_standard_factors['typical_price'][0]
roi = np.squeeze(np.asarray(test_df.as_matrix(['prev_roi_1']))) * test_standard_factors['prev_roi_%s' % delta][1] + test_standard_factors['prev_roi_%s' % delta][0]

# train 3 classifiers
clf1 = KNeighborsClassifier(n_neighbors=400)
clf1.fit(X_train, y_train)
clf2 = SVC(gamma=0.01, C=46.4158883336127772)
clf2.fit(X_train, y_train)
clf3 = LogisticRegression()
clf3.fit(X_train, y_train)
regr1 = LinearRegression()
regr1.fit(X_train, y_train)

# initialize 3 traders
money1 = {'ETH': 400, 'BTC':10}
money2 = {'ETH': 400, 'BTC':10}
money3 = {'ETH': 400, 'BTC':10}
money4 = {'ETH': 400, 'BTC':10}
all_in_BTC_1 = [money1['BTC'] + money1['ETH'] * typical_prices_test[0]]  # this is a list that will contain the conversion of all money in bitcoin, at each trading step, for trader 1
all_in_BTC_2 = [money2['BTC'] + money2['ETH'] * typical_prices_test[0]]  # idem for trader 2
all_in_BTC_3 = [money3['BTC'] + money3['ETH'] * typical_prices_test[0]]  # idem for trader 3
all_in_BTC_4 = [money4['BTC'] + money4['ETH'] * typical_prices_test[0]]  # idem for trader 4
ETH_1 = [money1['ETH']]
ETH_2 = [money2['ETH']]
ETH_3 = [money3['ETH']]
ETH_4 = [money4['ETH']]
BTC_1 = [money1['BTC']]
BTC_2 = [money2['BTC']]
BTC_3 = [money3['BTC']]
BTC_4 = [money4['BTC']]

has_bought_1 = False  # tells you whether
has_bought_2 = False
has_bought_3 = False
has_bought_4 = False
for k in range(X_test.shape[0]):
    data = X_test[k, :]
    prediction1 = clf1.predict(data.reshape(1, -1))[0]
    prediction2 = clf2.predict(data.reshape(1, -1))[0]
    prediction3 = np.sign(regr1.predict(data.reshape(1, -1))[0] * train_standard_factors['next_roi_%s'%delta][1] + train_standard_factors['next_roi_%s'%delta][0])
    prediction4 = clf3.predict(data.reshape(1, -1))[0]

    # TRADER 1
    if has_bought_1:
        if roi[k] > 0:
            if money1['ETH'] >= 1:
                money1['ETH'] -= 1
                money1['BTC'] += typical_prices_test[k]
        else:
            if prediction1 > 0:
                if money1['BTC'] >= typical_prices_test[k]:
                    money1['ETH'] += 1
                    money1['BTC'] -= typical_prices_test[k]
            else:
                if money1['ETH'] >= 1:
                    money1['ETH'] -= 1
                    money1['BTC'] += typical_prices_test[k]
        has_bought_1 = False
    if prediction1 > 0:
        if money1['BTC'] >= typical_prices_test[k]:
            money1['ETH'] += 1
            money1['BTC'] -= typical_prices_test[k]
            has_bought_1 = True
    else:
        if money1['ETH'] >= 0:
            money1['ETH'] -= 1
            money1['BTC'] += typical_prices_test[k]

    # TRADER 2
    if has_bought_2:
        if roi[k] > 0:
            if money2['ETH'] >= 1:
                money2['ETH'] -= 1
                money2['BTC'] += typical_prices_test[k]
        else:
            if prediction2 > 0:
                if money2['BTC'] >= typical_prices_test[k]:
                    money2['ETH'] += 1
                    money2['BTC'] -= typical_prices_test[k]
            else:
                if money2['ETH'] >= 1:
                    money2['ETH'] -= 1
                    money2['BTC'] += typical_prices_test[k]
        has_bought_2 = False
    if prediction2 > 0:
        if money2['BTC'] >= typical_prices_test[k]:
            money2['ETH'] += 1
            money2['BTC'] -= typical_prices_test[k]
            has_bought_2 = True
    else:
        if money2['ETH'] >= 1:
            money2['ETH'] -= 1
            money2['BTC'] += typical_prices_test[k]

    # TRADER 3
    if has_bought_3:
        if roi[k] > 0:
            if money3['ETH'] >= 1:
                money3['ETH'] -= 1
                money3['BTC'] += typical_prices_test[k]
        else:
            if prediction3 > 0:
                if money3['BTC'] >= typical_prices_test[k]:
                    money3['ETH'] += 1
                    money3['BTC'] -= typical_prices_test[k]
            else:
                if money3['ETH'] >= 1:
                    money3['ETH'] -= 1
                    money3['BTC'] += typical_prices_test[k]
        has_bought_3 = False
    if prediction3 > 0:
        if money3['BTC'] >= typical_prices_test[k]:
            money3['ETH'] += 1
            money3['BTC'] -= typical_prices_test[k]
            has_bought_3 = True
    else:
        if money3['ETH'] >= 1:
            money3['ETH'] -= 1
            money3['BTC'] += typical_prices_test[k]

    # TRADER 4
    if has_bought_4:
        if roi[k] > 0:
            if money4['ETH'] >= 10:
                money4['ETH'] -= 10
                money4['BTC'] += 10*typical_prices_test[k]
        else:
            if at_least_one_bigger(clf3.predict_proba(data.reshape(1, -1)), 0.65):
                if prediction4 > 0:
                    if money4['BTC'] >= 10*typical_prices_test[k]:
                        money4['ETH'] += 10
                        money4['BTC'] -= 10*typical_prices_test[k]
                else:
                    if money4['ETH'] >= 10:
                        money4['ETH'] -= 10
                        money4['BTC'] += 10*typical_prices_test[k]
        has_bought_4 = False
    if at_least_one_bigger(clf3.predict_proba(data.reshape(1, -1)), 0.65):
        if prediction4 > 0:
            if money4['BTC'] >= 10*typical_prices_test[k]:
                money4['ETH'] += 10
                money4['BTC'] -= 10*typical_prices_test[k]
                has_bought_4 = True
        else:
            if money4['ETH'] >= 10:
                money4['ETH'] -= 10
                money4['BTC'] += 10*typical_prices_test[k]

    # update bitcoin capital of each trader
    all_in_BTC_1.append(money1['BTC'] + money1['ETH'] * typical_prices_test[k])
    all_in_BTC_2.append(money2['BTC'] + money2['ETH'] * typical_prices_test[k])
    all_in_BTC_3.append(money3['BTC'] + money3['ETH'] * typical_prices_test[k])
    all_in_BTC_4.append(money4['BTC'] + money4['ETH'] * typical_prices_test[k])
    ETH_1.append(money1['ETH'])
    ETH_2.append(money2['ETH'])
    ETH_3.append(money3['ETH'])
    ETH_4.append(money4['ETH'])
    BTC_1.append(money1['BTC'])
    BTC_2.append(money2['BTC'])
    BTC_3.append(money3['BTC'])
    BTC_4.append(money4['BTC'])


f, axarr = plt.subplots(2, 2)
f.suptitle('Trader SVM (blue) vs Trader KNN (red) vs Linear regr (green) vs LR (magenta)')

axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_1)/all_in_BTC_1[0], c='r')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_2)/all_in_BTC_2[0], c='b')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_3)/all_in_BTC_3[0], c='g')
axarr[0,0].plot([k for k in range(len(all_in_BTC_1))], -1 + np.array(all_in_BTC_4)/all_in_BTC_4[0], c='m')
axarr[0,0].set_title("Overall BTC profit")

axarr[0,1].plot([k for k in range(len(all_in_BTC_1))], np.array(BTC_1)/np.array(all_in_BTC_1), c='r')
axarr[0,1].plot([k for k in range(len(all_in_BTC_1))], np.array(BTC_2)/np.array(all_in_BTC_2), c='b')
axarr[0,1].plot([k for k in range(len(all_in_BTC_1))], np.array(BTC_3)/np.array(all_in_BTC_3), c='g')
axarr[0,1].plot([k for k in range(len(all_in_BTC_1))], np.array(BTC_4)/np.array(all_in_BTC_4), c='m')
axarr[0,1].set_title("BTC liquidity as a percentage of the total")

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