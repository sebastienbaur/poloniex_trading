from functools import partial
import numpy as np


def macd(values, alpha=0.8, period1=26, period2=12):
    x = np.array(exponential_moving_average(values, alpha=alpha, period=period1))
    y = np.array(exponential_moving_average(values, alpha=alpha, period=period2))
    return x-y


def rsi(values, alpha=0.5):
    increases = [0]
    decreases = [0]
    for k in range(1, len(values)):
        if values[k] - values[k-1] > 0:
            increases.append(values[k] - values[k-1])
            decreases.append(0)
        else:
            decreases.append(values[k-1] - values[k])
            increases.append(0)
    increases_exp_avg = np.array(exponential_moving_average(increases, alpha=alpha))
    decreases_exp_avg = np.array(exponential_moving_average(decreases, alpha=alpha))
    return (100*(decreases_exp_avg / (decreases_exp_avg + increases_exp_avg))).tolist()


def next_roi(values, delta=1):
    results = []
    for k in range(0, len(values)-delta):
        result = (values[k+delta] - values[k])/values[k]
        results.append(result)
    return results+delta*[np.nan]


def prev_roi(values, delta=1):
    results = []
    for k in range(delta, len(values)):
        result = (values[k]-values[k-delta])/values[k-delta]
        results.append(result)
    return delta*[np.nan]+results


def prev_typical_price(high, low, close, period):
    typical_prices = typical_price(high, low, close)
    return period * [np.nan] + typical_prices[:-period]


def next_typical_price(high, low, close, period):
    typical_prices = typical_price(high, low, close)
    return typical_prices[period:] + period * [np.nan]


def typical_price(high, low, close):
    return ((np.array(high) + np.array(low) + np.array(close))/3).tolist()


def exponential_moving_average(values, period=None, alpha=0.5):
    if period is None:
        moving_averages = [alpha*values[0]]
        for k in range(1, len(values)):
            average = moving_averages[-1]*(1-alpha) + alpha*values[k]
            moving_averages.append(average)
        return moving_averages
    else:
        moving_averages = []
        for k in range(period, len(values)):
            moving_average = 0
            for j in range(k-period, k+1):
                moving_average += alpha*values[j]*((1-alpha)**period-j)
            moving_averages.append(moving_average)
        return period*[np.nan] + moving_averages


def moving_average(values, period):
    moving_averages = [values[0]]
    for k in range(1, len(values)):
        average = np.mean(values[k-min(k, period):k])
        moving_averages.append(average)
    return moving_averages


def volatility(values, period):
    volatilities = [np.nan]
    for k in range(1, len(values)):
        vol = np.var(values[k-min(k, period):k])
        volatilities.append(vol)
    return volatilities


def p_moment(values, period, p):
    moments = [np.nan]
    for k in range(1, len(values)):
        moment = np.mean(np.absolute(np.array(values[k-min(k, period):k]) - np.mean(values[k-min(k, period):k]))**p)
        moments.append(moment)
    return moments


def trix(values, alpha=0.5):
    triple_average = exponential_moving_average(exponential_moving_average(exponential_moving_average(values, alpha=alpha), alpha=alpha), alpha=alpha)
    results = [np.nan]
    for k in range(1, len(triple_average)):
        result = (triple_average[k] - triple_average[k-1]) / ((triple_average[k] + triple_average[k-1])/2)
        results.append(result)
    return results


def percent_r(high, low, close, period):
    rs = [np.nan]
    highest = high[0]
    lowest = low[0]
    close_price = close[0]
    high_prices =[high[0]]
    low_prices = [low[0]]
    for k in range(1, len(low)):
        if len(high_prices) < period:
            high_prices.append(high[k])
            low_prices.append(low[k])
        elif len(high_prices) == period:
            high_prices.pop()
            low_prices.pop()
            high_prices.append(high[k])
            low_prices.append(low[k])
        close_price = close[k]
        highest = max(high_prices)
        lowest = min(low_prices)
        r = (-100.0) * (highest - close_price) / (highest - lowest)
        rs.append(r)
    return rs


def avg(high, low, close, period):
    bps = []
    trs = []
    results = []
    for k in range(1, len(high)):
        bp = close[k] - min(low[k], close[k-1])
        tr = max(high[k], close[k-1]) - min(low[k], close[k-1])
        bps.append(bp)
        trs.append(tr)
        if len(bps) >= period:
            results.append(sum(bps[k-period:k])/sum(trs[k-period:k]))
        else:
            results.append(sum(bps[0:k])/sum(trs[0:k]))
    return [np.nan]+results


def ult_osc(high, low, close):
    avg7 = np.array(avg(high, low, close, 7))
    avg14 = np.array(avg(high, low, close, 14))
    avg28 = np.array(avg(high, low, close, 28))
    return [np.nan] + ((4*avg7 + 2*avg14 + avg28)/7).tolist()[1:]


def cci(typical_price, period, high=None, low=None, close=None):
    # values = typical_price(high, low, close)
    cci_values = ((np.array(typical_price) - np.array(moving_average(typical_price, period))) / np.sqrt(np.array(volatility(typical_price, period)))).tolist()
    for k, value in enumerate(cci_values):
        if value == np.inf or value == -np.inf:
            cci_values = cci_values[:k] + [np.nan] + cci_values[k+1:]
    return cci_values


SMALL_PERIOD = 5
BIG_PERIOD = 30

ALL_TYPICAL_FUNCTIONS = [
    partial(next_roi, delta=1),
    partial(prev_roi, delta=1),
    partial(prev_roi, delta=2),
    partial(prev_roi, delta=3),
    partial(prev_roi, delta=4),
    partial(prev_roi, delta=5),
    exponential_moving_average,
    partial(exponential_moving_average, period=SMALL_PERIOD),
    partial(exponential_moving_average, period=BIG_PERIOD),
    partial(moving_average, period=SMALL_PERIOD),
    partial(moving_average, period=BIG_PERIOD),
    partial(volatility, period=SMALL_PERIOD),
    partial(volatility, period=BIG_PERIOD),
    partial(p_moment, period=SMALL_PERIOD, p=3),
    partial(p_moment, period=BIG_PERIOD, p=3),
    partial(p_moment, period=SMALL_PERIOD, p=4),
    partial(p_moment, period=BIG_PERIOD, p=4),
    trix,
    partial(rsi, alpha=0.5),
    macd,
    partial(cci, period=SMALL_PERIOD),
    partial(cci, period=BIG_PERIOD)
]
ALL_HLC_FUNCTIONS = [
    typical_price,
    partial(prev_typical_price, period=1),
    partial(prev_typical_price, period=2),
    partial(prev_typical_price, period=3),
    partial(prev_typical_price, period=4),
    partial(prev_typical_price, period=5),
    partial(percent_r, period=SMALL_PERIOD),
    partial(percent_r, period=BIG_PERIOD),
    ult_osc
]
ALL_TYPICAL_NAMES = [
    'next_roi_1',
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
    'macd',
    'cci_%s' % SMALL_PERIOD,
    'cci_%s' % BIG_PERIOD
]
ALL_HLC_NAMES = [
    'typical_price',
    'prev_typical_price_1',
    'prev_typical_price_2',
    'prev_typical_price_3',
    'prev_typical_price_4',
    'prev_typical_price_5',
    'percent_r_%s' % SMALL_PERIOD,
    'percent_r_%s' % BIG_PERIOD,
    'ult_osc'
]
