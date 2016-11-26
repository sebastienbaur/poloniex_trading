from projet.lib.data_functions.utils import *

###############################
# GET DATA
# #############################
market = PoloniexMarket()
start_date = '2015/11/01'
end_date = '2016/06/01'
pair = "BTC_ETH"
## LOAD ...
# market_data = pd.read_csv('market_data %s %s.csv' % (start_date.replace('/', '-'), end_date.replace('/','-')), sep=';', index_col=0, header=0)

## ... OR CREATE AND ...
market_data = market.get_historical_data(pair, datestring_to_timestamp(start_date), end=datestring_to_timestamp(end_date), period=7200)
## ... SAVE
# market_data.to_csv('market_data %s %s.csv' % (start_date.replace('/', '-'), end_date.replace('/', '-')), sep=';')

df, scaling_factors = train_predictors(market_data, ALL_TYPICAL_FUNCTIONS, ALL_HLC_FUNCTIONS, ALL_TYPICAL_NAMES, ALL_HLC_NAMES)

roi = np.squeeze(np.asarray(df.as_matrix(['next_roi_1'])))[:-1]
fig1 = plt.figure(1)
plt.title("Distribution of scaled ROIs")
plt.hist(roi, bins=100)
fig1.show()

fig2 = plt.figure(2)
stats.probplot(roi, dist="norm", plot=plt)
fig2.show()

unscaled_roi = roi * scaling_factors["next_roi_1"][1] + scaling_factors["next_roi_1"][0]
fig3 = plt.figure(3)
plt.title("Distribution of unscaled ROIs")
plt.hist(unscaled_roi, bins=100)
fig3.show()

count = 0
for x in unscaled_roi:
    if -0.0025 < x < 0.0025:
        count+=1
print count, count*1.0/unscaled_roi.shape[0]