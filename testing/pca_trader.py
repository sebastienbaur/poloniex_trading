from itertools import product

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from projet.lib.data_functions.utils import *


def which_pca_zone(x, principal_components, percentiles):
    """
    :param x: a vector to be projected on the 3 first principal components
    :param principal_components: a list of the 3 first principal components
    :param percentiles: a list of 3 lists. The kth sublist contains the 25th, 50th and 75th percentiles of the kth principal component
    :return: a tuple x,y,z where each of the 3 components is part of [0,1,2,3]
    """
    proj1 = np.dot(x, principal_components[0])
    proj2 = np.dot(x, principal_components[1])
    proj3 = np.dot(x, principal_components[2])

    # loc1
    if proj1 < percentiles[0][0]:
        loc1=0
    elif percentiles[0][0] <= proj1 < percentiles[0][1]:
        loc1=1
    elif percentiles[0][1] <= proj1 < percentiles[0][2]:
        loc1=2
    else:
        loc1=3

    # loc2
    if proj2 < percentiles[1][0]:
        loc2=0
    elif percentiles[1][0] <= proj2 < percentiles[1][1]:
        loc2=1
    elif percentiles[1][1] <= proj2 < percentiles[1][2]:
        loc2=2
    else:
        loc2=3

    # loc3
    if proj3 < percentiles[2][0]:
        loc3=0
    elif percentiles[2][0] <= proj3 < percentiles[2][1]:
        loc3=1
    elif percentiles[2][1] <= proj3 < percentiles[2][2]:
        loc3=2
    else:
        loc3=3

    return loc1, loc2, loc3


def scores_to_csv(filename, scores, names, areas, data_per_area):
    f = open(filename, 'w')
    f.write('areas;')
    for name in names:
        f.write(name)
        f.write(';')
    f.write('\n')
    for area in areas:
        f.write(str(area))
        f.write(';')
        for name in names:
            f.write(str(scores[name][area]))
            f.write(';')
        f.write(str(data_per_area[area]))
        f.write('\n')
    f.close()


# train 2 classifiers
linear_regr = LinearRegression()
logistic_clf = LogisticRegression()
knn_1 = KNeighborsClassifier(n_neighbors=400)
knn_2 = KNeighborsClassifier(n_neighbors=100)
svm_1 = SVC(gamma=0.0083767764006829164, C=46.415888336127772)
svm_2 = SVC(gamma=0.010608183551394482, C=46.4158883336127772)
svm_3 = SVC(gamma=0.017012542798525893, C=0.21544346900318834)
svm_4 = SVC(gamma=0.088862381627434026, C=0.1)
svm_5 = SVC(gamma=0.043754793750741844, C=0.1)
tree = AdaBoostClassifier(n_estimators=1000, learning_rate=0.01)
pca = PCA(n_components=3)

pca.fit(X_train)
linear_regr.fit(X_train, y_train)
logistic_clf.fit(X_train, y_train)
knn_1.fit(X_train, y_train)
knn_2.fit(X_train, y_train)
svm_1.fit(X_train, y_train)
svm_2.fit(X_train, y_train)
svm_3.fit(X_train, y_train)
svm_4.fit(X_train, y_train)
svm_5.fit(X_train, y_train)
tree.fit(X_train, y_train)
names = ['knn_1', 'knn_2', 'svm_1', 'svm_2', 'svm_3', 'svm_4', 'svm_5', 'linear_regr', 'logistic_clf', 'tree']

first_component = pca.components_[0]
second_component = pca.components_[1]
third_component = pca.components_[2]
projections_3d = np.array([[np.dot(X_train[k, :], first_component), np.dot(X_train[k, :], second_component), np.dot(X_train[k, :], third_component)] for k in range(X_train.shape[0])])
first_component_percentiles = np.percentile(projections_3d[:, 0], [25, 50, 75]).tolist()
second_component_percentiles = np.percentile(projections_3d[:, 0], [25, 50, 75]).tolist()
third_component_percentiles = np.percentile(projections_3d[:, 0], [25, 50, 75]).tolist()

all_areas = [x for x in product([0,1,2,3], [0,1,2,3], [0,1,2,3])]

# initialize 2 traders
money = {'ETH': 400, 'BTC': 10}
all_in_BTC = [money['BTC'] + money['ETH'] * typical_prices_test[0]]  # this is a list that will contain the conversion of all money in bitcoin, at each trading step
BTC = [money['BTC']]
ETH = [money['ETH']]

scores = {name: {area: 1 for area in all_areas} for name in names}
data_per_area = {area: 0 for area in all_areas}
has_bought = False  # tells you whether
for k in range(X_test.shape[0]):
    if k > 1:
        for pred, name in zip(predictions, names):
            if pred*roi[k] > 0:
                scores[name][area] += 1
    data = X_test[k, :]
    prediction1 = knn_1.predict(data.reshape(1, -1))[0]
    prediction2 = knn_2.predict(data.reshape(1, -1))[0]
    prediction3 = svm_1.predict(data.reshape(1, -1))[0]
    prediction4 = svm_2.predict(data.reshape(1, -1))[0]
    prediction5 = svm_3.predict(data.reshape(1, -1))[0]
    prediction6 = svm_4.predict(data.reshape(1, -1))[0]
    prediction7 = svm_5.predict(data.reshape(1, -1))[0]
    prediction8 = np.sign(linear_regr.predict(data.reshape(1, -1))[0] * scaling_factors['next_roi_%s'%delta][1] + scaling_factors['next_roi_%s'%delta][0])
    prediction9 = logistic_clf.predict(data.reshape(1, -1))[0]
    prediction10 = tree.predict(data.reshape(1, -1))[0]
    area = which_pca_zone(data, [first_component, second_component, third_component], [first_component_percentiles, second_component_percentiles, third_component_percentiles])
    data_per_area[area] += 1
    predictions = [prediction1, prediction2, prediction3, prediction4, prediction5, prediction6, prediction7, prediction8, prediction9, prediction10]
    prediction = np.sum(np.array(predictions)*np.array([scores[name][area] for name in names])) / np.sum(np.array([scores[name][area] for name in names]))
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
    BTC.append(money['BTC'])
    ETH.append(money['ETH'])


print money
print money['BTC'] + money['ETH']*typical_prices_test[-1]
scores_to_csv('scores4.csv', scores, names, all_areas, data_per_area)


f, axarr = plt.subplots(2, 2)
f.suptitle('Multiheaded PCA trader')

axarr[0,0].plot([k for k in range(len(all_in_BTC))], -1 + np.array(all_in_BTC)/all_in_BTC[0], c='r')
axarr[0,0].set_title("Overall BTC profit")

axarr[0,1].plot([k for k in range(len(BTC))], np.array(BTC)/np.array(all_in_BTC), c='b')
axarr[0,1].set_title("BTC liquidity as a percentage of the total")

axarr[1,0].plot([k for k in range(len(BTC))], -1 + np.array(BTC)/BTC[0], c='g')
axarr[1,0].set_title("BTC profit as a percentage of the initial value")

axarr[1,1].plot([k for k in range(len(ETH))], -1 + np.array(ETH)/ETH[0], c='m')
axarr[1,1].set_title("ETH profit as a percentage of the initial value")

f.show()

