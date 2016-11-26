from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from projet.lib.data_functions.utils import *

# train 2 classifiers
knn_1 = KNeighborsClassifier(n_neighbors=400)
knn_2 = KNeighborsClassifier(n_neighbors=100)
svm_1 = SVC(gamma=0.0083767764006829164, C=46.415888336127772)
svm_2 = SVC(gamma=0.010608183551394482, C=46.4158883336127772)
svm_3 = SVC(gamma=0.017012542798525893, C=0.21544346900318834)
svm_4 = SVC(gamma=0.088862381627434026, C=0.1)
svm_5 = SVC(gamma=0.043754793750741844, C=0.1)
# pca = PCA(n_components=3)

# pca.fit(X_train)
knn_1.fit(X_train, y_train)
knn_2.fit(X_train, y_train)
svm_1.fit(X_train, y_train)
svm_2.fit(X_train, y_train)
svm_3.fit(X_train, y_train)
svm_4.fit(X_train, y_train)
svm_5.fit(X_train, y_train)
names = ['knn_1', 'knn_2', 'svm_1', 'svm_2', 'svm_3', 'svm_4', 'svm_5']

# initialize 2 traders
money = {'ETH': 400, 'BTC':10}
all_in_BTC = [money['BTC'] + money['ETH'] * typical_prices_test[0]]  # this is a list that will contain the conversion of all money in bitcoin, at each trading step
BTC = [money['BTC']]
ETH = [money['ETH']]

scores = {name:1 for name in names}
has_bought = False  # tells you whether
for k in range(X_test.shape[0]):
    if k > 1:
        for pred, name in zip(predictions, names):
            if pred*roi[k] > 0:
                scores[name] += 1
    data = X_test[k, :]
    prediction1 = knn_1.predict(data.reshape(1, -1))[0]
    prediction2 = knn_2.predict(data.reshape(1, -1))[0]
    prediction3 = svm_1.predict(data.reshape(1, -1))[0]
    prediction4 = svm_2.predict(data.reshape(1, -1))[0]
    prediction5 = svm_3.predict(data.reshape(1, -1))[0]
    prediction6 = svm_4.predict(data.reshape(1, -1))[0]
    prediction7 = svm_5.predict(data.reshape(1, -1))[0]
    predictions = [prediction1, prediction2, prediction3, prediction4, prediction5, prediction6, prediction7]
    prediction = np.sum(np.array(predictions)*np.array([scores[name] for name in names])) / np.sum(np.array([scores[name] for name in names]))
    if has_bought:
        if roi[k] > 0:
            if money['ETH'] >= 1:
                money['ETH'] -= 1
                money['BTC'] += typical_prices_test[k]
        else:
            if prediction > 0:
                if money['BTC'] >= typical_prices_test[k]:
                    money['ETH'] += 1
                    money['BTC'] -= typical_prices_test[k]
            else:
                if money['ETH'] >= 1:
                    money['ETH'] -= 1
                    money['BTC'] += typical_prices_test[k]
        has_bought = False
    if prediction > 0:
        if money['BTC'] >= typical_prices_test[k]:
            money['ETH'] += 1
            money['BTC'] -= typical_prices_test[k]
            has_bought = True
    else:
        if money['ETH'] >= 1:
            money['ETH'] -= 1
            money['BTC'] += typical_prices_test[k]
    # update bitcoin capital
    all_in_BTC.append(money['BTC'] + money['ETH'] * typical_prices_test[k])
    ETH.append(money['ETH'])
    BTC.append(money['BTC'])


print money
print "All in Bitcoin : %s" % money['BTC'] + money['ETH']*typical_prices_test[-1]


f, axarr = plt.subplots(2, 2)
f.suptitle('Multiheaded trader')

axarr[0,0].plot([k for k in range(len(all_in_BTC))], -1 + np.array(all_in_BTC)/all_in_BTC[0], c='r')
axarr[0,0].set_title("Overall BTC profit")

axarr[0,1].plot([k for k in range(len(BTC))], np.array(BTC)/np.array(all_in_BTC), c='b')
axarr[0,1].set_title("BTC liquidity as a percentage of the total")

axarr[1,0].plot([k for k in range(len(BTC))], -1 + np.array(BTC)/BTC[0], c='g')
axarr[1,0].set_title("BTC profit as a percentage of the initial value")

axarr[1,1].plot([k for k in range(len(ETH))], -1 + np.array(ETH)/ETH[0], c='m')
axarr[1,1].set_title("ETH profit as a percentage of the initial value")

f.show()