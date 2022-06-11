# EDA method to data analysis

"""
1. Loading the necessary libraries and setting them up
2. Data collection
3. Data wrangling/munging
4. Data cleaning
5. Obtaining descriptive statistics
6. Visual inspection of the data
7. Data cleaning
8. Advanced visualization techniques

"""

# 1 - Loading libraries
"""
We will be using numpy, pandas, and matplotlib, and these libraries can be loaded with the
help of the following code:
"""
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt
import mpld3 # We use the mpld3 library for enabling zooming within Jupyter's matplotlib charts.
# mpld3.enable_notebook()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 2 - Data collection
DJIA_data = pd.read_csv('^DJI_data.csv', parse_dates=True, index_col=0)
print('DJIA_data \n', DJIA_data.describe())
SPX_data = pd.read_csv('^GSPC_data.csv', parse_dates=True, index_col=0)
print('SPX_data \n', SPX_data.describe())
VIX_data = pd.read_csv('^VIX_data.csv', parse_dates=True, index_col=0)
print('VIX_data \n', VIX_data.describe())

# 3 - Data wrangling/munging
"""
Data rarely comes in a ready-to-use format. Data wrangling/munging refers to the process of manipulating and 
transforming data from its initial raw source into structured, formatted, and easily usable datasets.
"""

merged_df = DJIA_data.join(SPX_data, how='outer', lsuffix='_DJIA', sort=True).\
    join(VIX_data, how='outer', lsuffix='_SPX', rsuffix='_VIX', sort=True)
print('Financial data \n', merged_df.tail())

# 4 & 5 - Data cleaning & Description
"""
Data cleaning refers to the process of addressing data errors coming from missing data, incorrect data values, 
and outliers.
"""

# 4.1 - Let's first check if there are any rows where all values are missing (NaN), as follows:
pd.set_option('display.max_rows', 5)
df_Is_Null = merged_df[merged_df.isnull().all(axis=1)]
print('df_Is_Null \n', df_Is_Null)

# 4.2 - Now, let's find out how many rows exist that have at least one field that is missing/NaN, as follows:
print('Close price is null \n', merged_df[['Close_DJIA', 'Close_SPX', 'Close_VIX']].isnull().any(axis=1).sum())

# 4.3 - Check for NaN on Close Price column
valid_close_df = merged_df.dropna(subset=['Close_DJIA', 'Close_SPX', 'Close_VIX'], how='any')
print('Df: valid close \n', valid_close_df.describe())

# 4.4 - After dropping the missing Close prices, we should have no more missing Close price fields, as illustrated
# in the following code snippet:
print(valid_close_df[['Close_DJIA', 'Close_SPX', 'Close_VIX']].isnull().any(axis=1).sum())

# 4.5. - Checking for NaN for all other fields
"""
Next, let's deal with rows that have NaN values for any of the other fields, starting with getting a sense of how many 
such rows exist. We can do this by running the following code:
"""
print(valid_close_df.isnull().any(axis=1).sum())

# 4.6 - Let's quickly inspect a few of the rows with at least some fields with a missing value, as follows:
print(valid_close_df[valid_close_df.isnull().any(axis=1)])

# 4.7. - Fill NaN column kind of method
"""
Let's use the pandas.DataFrame.fillna(...) method with a method called backfill—this uses the next valid value after 
the missing value to fill in the missing value. The code is illustrated in the following snippet:
"""
valid_close_complete = valid_close_df.fillna(method='backfill')

"""
Let's see the impact of the back-filling, as follows:
"""
valid_close_complete.isnull().any(axis=1).sum()

# 4.8 - Obtaining descriptive statistics
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(valid_close_complete.describe())

# 4.9 - Removing the data that are not needed such as volume, open, low and high
pd.set_option('display.max_rows', 2)
prices_only = valid_close_complete.drop(['Volume_DJIA', 'Volume_SPX', 'Volume_VIX'], axis=1)
print('prices only \n', prices_only)

# 6 - Visual inspection of the data
"""  OHLC data visualization"""
tickers = ['DJIA', 'SPX', 'VIX']
for ticker in tickers:
    valid_close_complete['Open_'+str(ticker)].plot(figsize=(12, 6), linestyle='--', color='black',
                                                   legend='Open_'+str(ticker))
    valid_close_complete['Close_'+str(ticker)].plot(figsize=(12, 6), linestyle='-', color='grey',
                                                    legend='Close_'+str(ticker))
    valid_close_complete['Low_'+str(ticker)].plot(figsize=(12, 6), linestyle=':', color='black',
                                                  legend='Low_'+str(ticker))
    valid_close_complete['High_'+str(ticker)].plot(figsize=(12, 6), linestyle='-.', color='grey',
                                                   legend='High_'+str(ticker))
    plt.title('OHLC '+str(ticker))
    plt.savefig('OHLC '+str(ticker)+'.png')
    plt.show()

# 7 - Outliers
# 7.1 - IQR
"""
The two most commonly used methods to detect and remove outliers are the inter-quartile range (IQR) and the Z-score.
IQR
The IQR method uses a percentile/quantile range of values over the entire dataset to identify and remove outliers.
When applying the IQR method, we usually use extreme percentile values, such as 5% to 95%, to minimize the risk of 
removing correct data points.
"""

# 7.2 - Z-score
"""
Z-score
The Z-score (or standard score) is obtained by subtracting the mean of the dataset from each data point and normalizing 
the result by dividing by the standard deviation of the dataset.
In other words, the Z-score of a data point represents the distance in the number of standard deviations that the data 
point is away from the mean of all the data points.
"""

"""
First, we use scipy.stats.zscore(...) to compute Z-scores of each column in the prices_only DataFrame, and then we use 
numpy.abs(...) to get the magnitude of the Z-scores. Finally, we select rows where all fields have Z-scores lower than 
6, and save that in a no_outlier_prices 
DataFrame. The code is illustrated in the following snippet:
"""
no_outlier_prices = prices_only[(np.abs(stats.zscore(prices_only)) < 6).all(axis=1)]
print('no outlier prices \n',no_outlier_prices.describe())

# 7.3 - let's plot the no outlier prices

for ticker in tickers:
    no_outlier_prices['Open_'+str(ticker)].plot(figsize=(12, 6), linestyle='--', color='black',
                                                legend='Open_'+str(ticker))
    no_outlier_prices['Close_'+str(ticker)].plot(figsize=(12, 6), linestyle='-', color='grey',
                                                legend='Close_'+str(ticker))
    no_outlier_prices['Low_'+str(ticker)].plot(figsize=(12, 6), linestyle=':', color='black',
                                                legend='Low_'+str(ticker))
    no_outlier_prices['High_'+str(ticker)].plot(figsize=(12, 6), linestyle='-.', color='grey',
                                                legend='High_'+str(ticker))
    plt.title('No outlier OHLC '+str(ticker))
    plt.savefig('No outlier OHLC '+str(ticker)+'.png')
    plt.show()

# 7.4 - Description of data after removing outliers
"""
Let's also check the impact of our outlier removal work by re-inspecting the descriptive statistics, as follows:
"""

pd.set_option('display.max_rows', None)

for ticker in tickers:
    print('no outliers ' + str(ticker) + '\n', no_outlier_prices[['Open_'+str(ticker),
                                                                  'Close_'+str(ticker),
                                                                  'Low_'+str(ticker),
                                                                  'High_'+str(ticker)]].describe())

"""
Let's reset back the number of rows to display for a pandas DataFrame, as follows:
"""
pd.set_option('display.max_rows', 5)

# 7.5 - Advanced visualization techniques
"""
Let us explore univariate and multivariate statistics visualization techniques.
First, we will collect the close price for the three instruments.
"""
close_prices = no_outlier_prices[['Close_DJIA', 'Close_SPX', 'Close_VIX']]

"""
Let us now compute the daily close price changes to evaluate if there is a relationship between daily price changes
between the three instruments.
"""
# 7.5.1 - Daily close price changes
delta_close_prices = (close_prices.shift(-1) - close_prices).fillna(0)
delta_close_prices.columns = ['Delta_Close_DJIA', 'Delta_Close_SPX', 'Delta_Close_VIX']
print('delta close prices \n', delta_close_prices)

# 7.5.2 - Statistics description
pd.set_option('display.max_rows', None)
print('statistics data for delta close \n', delta_close_prices.describe())

"""
We can observe from these statistics that all three delta values' means are close to 0.
"""

# 7.5.3. - Histogram plot
"""
Let's observe the distribution of Delta_Close_TSLA to get more familiar with it, using a histogram plot.
"""
delta_close_prices['Delta_Close_VIX'].plot(kind='hist', bins=100, figsize=(12, 6), color='black', grid=True)
title = 'Histogram of Delta_Close_%s values roughly normally distributed around the 0 value' % ('VIX')
plt.title(title)
plt.savefig(title+'.png')
plt.show()
"""
Observation - we can see that the distribution is approximately normally distributed.
"""

# 7.5.4. - Box plot
"""
Let's draw a box plot, which also helps in assessing the values' distribution. The code for this is shown in the 
following snippet:
"""
delta_close_prices['Delta_Close_SPX'].plot(kind='box', figsize=(12,6), color='black', grid=True)
title = '%s Box plot showing mean, median, IQR (25th to 75th percentile), and outliers' % ('SPX')
plt.title(title)
plt.savefig(title+'.png')
plt.show()

# 7.5.5. - Correlation charts
"""
The first step in multivariate data statistics is to assess the correlations between Delta_Close_%s, Delta_Close_%s, 
and Delta_Close_%s. % ('DJIA', 'SPX', 'VIX')
The most convenient way to that is to plot a correlation scatter matrix, that shows that shows the pairwise relationship
between the three variables, as well as the distribution of each individual variable.
For this comparison, the KDE, kernel density estimation will be used.
"""
pd.plotting.scatter_matrix(delta_close_prices, figsize=(10, 10), color='black', alpha=0.75, diagonal='kde', grid=True)
title = 'Scatter plot of Delta_Close fields with KDE histogram on the diagonals'
plt.title(title)
plt.savefig(title+'.png')
plt.show()

# 7.5.6. - Statistics to check the correlation between variables
"""
Next, let's look at some statistics that provide the relationship between the variables. Correlation will allow us to
have it.
"""
print('Correlation matrix between stocks \n', delta_close_prices.corr())

# 7.5.7. - Pairwise correlation heatmap
"""
An alternative visualization technique known as a heatmap is available in seaborn.heatmap(...).
In the plot shown in the following screenshot, the rightmost scale shows a legend where the darkest values represent 
the strongest negative correlation and the lightest values represent the strongest positive correlations.
"""
plt.figure(figsize=(6, 6))
sn.heatmap(delta_close_prices.corr(), annot=True, square=True, linewidths=2)
title = 'Seaborn heatmap visualizing pairwise correlations between Delta_Close fields'
plt.title(title)
plt.savefig(title+'.png')
plt.show()

"""
# 8. - Special Python libraries for EDA
import dtale

import warnings
warnings.filterwarnings('ignore')
dtale.show(valid_close_df)
"""