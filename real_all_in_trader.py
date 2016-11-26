from sklearn.ensemble import AdaBoostClassifier
from projet.lib.data_functions.config import *
from projet.lib.data_functions.data_parser import datetime_to_timestamp, typical_prices_from_trades_history, _get_biggest_anterior_date, predictors_from_typical_prices
from projet.lib.data_functions.utils import Pricer

# starting the pricer, that will get periodically the prices buyers and sellers are ready to pay
print "started"
pricer = Pricer()
pricer.start()
print "pricer running"

# account information to make requests to the poloniex db
secret = os.environ.get('POLONIEX_SECRET')
api_key = os.environ.get('POLONIEX_API_KEY')
kraken_btc_address = os.environ.get('KRAKEN_BTC_ADDRESS')
market = PoloniexMarket(api_key=api_key, secret=secret)
order_number = 0

# train the classifier that tells you whether the price will go up or down
print "began training"
clf = AdaBoostClassifier(n_estimators=1000, learning_rate=0.01)
clf.fit(X_train, y_train)
set = False
is_first_time = True
print "trained"

# begin trading
while True:
    # wait for the time to arrive at the next 5min time
    if not set:
        compteur = datetime_to_timestamp(datetime.datetime.now()) // 300
        set = True
    if compteur == datetime_to_timestamp(datetime.datetime.now()) // 300 - 1:
        # cancel previous order
        open_orders = market.my_open_orders(pair)
        print "\n #################################################################### \nopen orders : %s" % open_orders
        if len(open_orders) > 0:
            for order in open_orders:
                order_number = int(order["orderNumber"])
                print "canceled at %s : %s " % ((datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(), market.cancel_order(order_number))

        # get and parse data
        set = False
        now = _get_biggest_anterior_date(datetime.datetime.now())
        start = datetime_to_timestamp(now) - 30000  # date since when you want to get prices
        dates, typical_prices, volumes = typical_prices_from_trades_history(market.get_trade_history(pair, start))
        predictors = predictors_from_typical_prices(typical_prices, volumes, scaling_factors)
        data = predictors.as_matrix(X_columns)

        # predict and write current data to file so that it can be exploited later
        prediction = clf.predict(data[-1:])[0]
        print "prediction : %s" % prediction
        if is_first_time:
            last_prediction = 1.0*prediction
        balance = market.balances()
        btc = float(balance['BTC'])
        eth = float(balance['ETH'])
        print "btc : %s" % btc
        print "eth : %s" % eth
        btc_amount = float(btc) + float(eth)*(typical_prices[-1])
        print "btc total value : %s" % btc_amount
        PoloniexMarket.wallet_to_csv(typical_prices, btc, eth, now, 'bif.csv', prediction, last_prediction, is_first_time)

        # according to your prediction, buy or sell
        if prediction > 0:
            if btc > 0:
                with Pricer.verrou:
                    print "buy order : %s" % market.buy(pair, Pricer.sell_price, btc/Pricer.sell_price, immediate_or_cancel=1, fill_or_kill=0, post_only=0)
        else:
            if eth > 0:
                with Pricer.verrou:
                    print "sell order : %s" % market.sell(pair, Pricer.buy_price, eth, immediate_or_cancel=1, fill_or_kill=0, post_only=0)
        # the first time requires a particular treatment. It's not the case afterwards
        if is_first_time:
            is_first_time = False
        # get the last prediction so that you know about your right/wrong decisions
        last_prediction = 1.0*prediction

