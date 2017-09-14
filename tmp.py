import numpy as np
import pandas as pd
# from pandas_datareader import data, wb
import pandas_datareader as pdr
import statsmodels as sm

# data wrangling - get Rates from yahoo using pandas
data = pdr.get_data_yahoo('^TNX', start='1,1,2017')  # ,end='1,1,2017')
data = data.drop(['Open', 'Close', 'Adj Close', 'Volume'], axis=1).dropna()
data['dly_range'] = np.absolute(np.diff(data.values, axis=1))
dta = pd.Series(data['dly_range'] * 0.627, index=data.index)

# Markov Switching Dynamic Regreassion / 3 regimes / Mean and Variance switching
ms3r = sm.MarkovRegression(dta, k_regimes=3, switching_variance=True)

np.random.seed(12345)
res_ms3r = ms3r.fit(search_reps=300)
print(res_ms3r.summary())

pbv0 = res_ms3r.smoothed_marginal_probabilities[0]
pbv1 = res_ms3r.smoothed_marginal_probabilities[1]
pbv2 = res_ms3r.smoothed_marginal_probabilities[2]

'Apex AD2600 Progressive-scan DVD player.txt',
'Canon G3.txt',
'Creative Labs Nomad Jukebox Zen Xtra 40GB.txt',
'Nikon coolpix 4300.txt',
'Nokia 6610.txt',
