import json

from data_parser import *
from indicators import SMALL_PERIOD, BIG_PERIOD
from projet.lib.data_functions.utils import PoloniexMarket

###############################
# GET DATA
# #############################
market = PoloniexMarket()
start_train_date = '2016/03/01'
end_train_date = '2016/05/01'
start_test_date = '2016/03/01'  # '2015/11/01'
end_test_date = '2016/06/28'  # '2016/03/01'
pair = "BTC_ETH"
## LOAD ...
# test_market_data = pd.read_csv('test_data %s %s.csv' % (start_test_date.replace('/', '-'), end_test_date.replace('/','-')), sep=';', index_col=0, header=0)
# train_market_data = pd.read_csv('train_data %s %s.csv' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-')), sep=';', index_col=0, header=0)

## ... OR CREATE AND ...
# train_market_data = market.get_historical_data(pair, datestring_to_timestamp(start_train_date), end=datestring_to_timestamp(end_train_date))
# test_market_data = market.get_historical_data(pair, datestring_to_timestamp(start_test_date), end=datestring_to_timestamp(end_test_date))
## ... SAVE
# train_market_data.to_csv('train_data %s %s.csv' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-')), sep=';')
# test_market_data.to_csv('test_data %s %s.csv' % (start_test_date.replace('/', '-'), end_test_date.replace('/', '-')), sep=';')

# train_df, scaling_factors = train_predictors(train_market_data, ALL_TYPICAL_FUNCTIONS, ALL_HLC_FUNCTIONS, ALL_TYPICAL_NAMES, ALL_HLC_NAMES)
# train_df = parse_predictor_dataframe(train_df)
# train_df = train_df.dropna()
# train_df.to_csv('train_predictors %s %s.csv' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-')), sep=';')
# with open('scaling_factors %s %s.json' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-')), 'w') as fp:
#     json.dump(scaling_factors, fp)

# LOAD DIRECTLY THE TRAIN PREDICTORS, SO THAT YOU DON'T RECOMPUTE THEM
train_df = pd.read_csv('train_predictors %s %s.csv' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-')), sep=';', header=0, index_col=0)
with open('scaling_factors %s %s.json' % (start_train_date.replace('/', '-'), end_train_date.replace('/','-'))) as data_file:
    scaling_factors = json.load(data_file)
# test_df = test_predictors(test_market_data, ALL_TYPICAL_FUNCTIONS, ALL_HLC_FUNCTIONS, ALL_TYPICAL_NAMES, ALL_HLC_NAMES, scaling_factors)
# test_df = parse_predictor_dataframe(test_df)
# test_df = test_df.dropna()


X_columns = [
    "volume",
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
    # 'percent_r_%s' % SMALL_PERIOD,
    # 'percent_r_%s' % BIG_PERIOD,
    # 'ult_osc',
    'macd',
    'cci_%s' % SMALL_PERIOD,
    'cci_%s' % BIG_PERIOD
]

delta = 1
X_train = train_df.as_matrix(X_columns)
y_train = np.squeeze(np.asarray(train_df.as_matrix(['next_roi_%s' % delta])))  # this is scaled
y_train = np.sign(y_train * scaling_factors['next_roi_%s' % delta][1] + scaling_factors['next_roi_%s' % delta][0])
# X_test = test_df.as_matrix(X_columns)
# y_test = np.squeeze(np.asarray(test_df.as_matrix(['next_roi_%s' % delta])))
# y_test = np.sign(y_test * scaling_factors['next_roi_%s' % delta][1] + scaling_factors['next_roi_%s' % delta][0])
# don't forget to unscale the variables
# typical_prices_test = np.squeeze(np.asarray(test_df.as_matrix(['typical_price']))) * scaling_factors['typical_price'][1] + scaling_factors['typical_price'][0]
# roi = np.squeeze(np.asarray(test_df.as_matrix(['prev_roi_%s' % delta]))) * scaling_factors['prev_roi_%s' % delta][1] + scaling_factors['prev_roi_%s' % delta][0]