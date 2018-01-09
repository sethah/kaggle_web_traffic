import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
%matplotlib inline

pd.options.display.max_rows = 10
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 600
from tqdm import tqdm
import gc

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import PCA

np.random.seed=0

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def smape_mask(y_true, y_pred, threshold):
    denominator = (np.abs(y_true) + np.abs(y_pred)) 
    diff = np.abs(y_true - y_pred) 
    diff[denominator == 0] = 0.0
    
    return diff <= (threshold / 2.0) * denominator
  
data_dir = "/home/cdsw/data/"
  
max_size = 184 # number of days in 2015

train_all = pd.read_csv(data_dir + "train_2.csv")

train_key = train_all[['Page']].copy()
all_page = train_all.Page.copy()

def get_date_index(date, train_all=train_all):
    for idx, c in enumerate(train_all.columns):
        if date == c:
            break
    if idx == len(train_all.columns):
        return None
    return idx
  
train_end = get_date_index('2016-09-10') + 1
test_start = get_date_index('2016-09-13')

train = train_all.iloc[ : , (train_end - max_size) : train_end].copy().astype('float32')
test = train_all.iloc[:, test_start : (63 + test_start)].copy().astype('float32')

train_all = train_all.iloc[:,-(max_size):].astype('float32')

train = train.iloc[:,::-1].copy()
train_all = train_all.iloc[:,::-1].copy()

data = [page.split('_') for page in tqdm(train_key.Page)]

access = ['_'.join(page[-2:]) for page in data]

site = [page[-3] for page in data]

page = ['_'.join(page[:-3]) for page in data]
page[:2]

train_key['PageTitle'] = page
train_key['Site'] = site
train_key['AccessAgent'] = access
train_key.head()

train_norm = np.log1p(train)
train_norm.head()

train_all_norm = np.log1p(train_all)

# explode the test data frame so each observation has a row
# then drop null observations
test['Page'] = all_page
test = pd.melt(test, id_vars=['Page'], var_name='Date', value_name='Visits_true')
test['Week'] = pd.to_datetime(test.Date).dt.dayofyear // 7
test = test.merge(train_key, how='left', on='Page')
test['Visits_true'] = test.Visits_true.astype('float32')
test['Visits_norm'] = np.log1p(test.Visits_true).astype('float32')
test = test[test.Visits_true.isnull() != True].reset_index(drop=True)

test.head()

# test_all is the actual test data we need to make predictions for
test_all = pd.read_csv(data_dir + 'key_2.csv')
test_all.head()

test_all['Date'] = [page[-10:] for page in tqdm(test_all.Page)]
test_all['Page'] = [page[:-11] for page in tqdm(test_all.Page)]
test_all.head()

test_all['Week'] = pd.to_datetime(test_all.Date).dt.dayofyear // 7
test_all = test_all.merge(train_key, how='left', on='Page')
test_all.head()

sites = train_key.Site.unique()
sites

# all visits is median

max_periods = 20
periods = [(0,1), (1,2), (2,3), (3,4), 
           (4,5), (5,6), (6,7), (7,8),
           (0,2), (0,4),
           (0,8), (0,12), 
           (0,16),
           (0, max_periods)
          ]

def add_median(test, train, train_key, periods, max_periods, first_train_weekday):
    train =  train.iloc[:,:7*max_periods]
    train_weekday = np.array([(first_train_weekday-w) % 7 for w in range(train.shape[1])])
    train_week_idx = [i for i,w in enumerate(train_weekday) if w <= 4]
    train_week = train.iloc[:,train_week_idx] # train set but with only weekdays
    train_weekend_idx = [i for i,w in enumerate(train_weekday) if w > 4]
    train_weekend = train.iloc[:,train_weekend_idx] # train set but with only weekends

    test_week = (test.WeekDay <= 4)
    test_weekend = ~test_week
    test['WeekEnd'] = 1 * test_weekend
    df = train_key[['Page']].copy()
    df['AllVisits'] = train.median(axis=1).fillna(0)
    test = test.merge(df, how='left', on='Page', copy=False)
    test.AllVisits = test.AllVisits.fillna(0).astype('float32')
    
    for (w1, w2) in tqdm(periods):
        
        df = train_key[['Page']].copy()
        c = 'median_%d_%d' % (w1, w2)
        df[c] = train.iloc[:,7*w1:7*w2].median(axis=1, skipna=True) 
        test = test.merge(df, how='left', on='Page', copy=False)
        test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')
        
        c = 'median_day_%d_%d' % (w1, w2)
        test_page = test[['Page']].copy()
        
        df = train_key[['Page']].copy()
        df[c] = train_week.iloc[:,5*w1:5*w2].median(axis=1, skipna=True) 
        df = test_page.loc[test_week].merge(df, how='left', on='Page', copy=False)
        test.loc[test_week, c] = df[c].values
        
        df = train_key[['Page']].copy()
        df[c] = train_weekend.iloc[:,2*w1:2*w2].median(axis=1, skipna=True) 
        df = test_page.loc[test_weekend].merge(df, how='left', on='Page', copy=False)
        test.loc[test_weekend, c] = df[c].values

        test[c] = (test[c] - test.AllVisits).fillna(0).astype('float32')

    gc.collect()

    return test
  
test0 = test.copy()
test_all0 = test_all.copy()

npca = 0
max_periods = 20
periods = [(0,1), (1,2), (2,3), (3,4), 
           (4,5), (5,6), (6,7), (7,8),
           (0,2), (0,4),
           (0,8), (0,12), 
           (0,16),
           (0, max_periods)
          ]

test, test_all = test0.copy(), test_all0.copy()

res = 0
res_den = 0
out = []

test['Visits'] = 0
test_all['Visits'] = 0

threshold = 1.25

for site in sites:
    print(site)

    train_norm_site = train_norm[train_key.Site == site]
    train_all_norm_site = train_all_norm[train_key.Site == site]
    train_key_site = train_key[train_key.Site == site]

    test_site = test[test.Site == site].reset_index(drop=True)
    test_site['Date'] = pd.to_datetime(test_site.Date)
    test_site['WeekDay'] = test_site.Date.dt.dayofweek
    
    test_all_site = test_all[test_all.Site == site].reset_index(drop=True)
    test_all_site['Date'] = pd.to_datetime(test_all_site.Date)
    test_all_site['WeekDay'] = test_all_site.Date.dt.dayofweek

    test1 = add_median(test_site, train_norm_site, train_key_site, periods, max_periods, 3)
    test_all1 = add_median(test_all_site, train_all_norm_site, train_key_site, periods, max_periods, 5)
    
    test1.Visits_norm -= test1.AllVisits
    
    num_cols = (['median_day_%d_%d' % (w1,w2) for (w1,w2) in periods]) 

    print('threshold: %0.2f' % threshold)
    res_site = 0
    res_site_den = 0
    for week in test_site.Week.unique():
        #print('week:', week)
        test2 = test1[test1.Week == week].reset_index(drop=True)
        
        lr = HuberRegressor(epsilon=1)
        
        lr.fit(test2[num_cols], test2.Visits_norm)
        y = lr.predict(test2[num_cols])
        y += test2.AllVisits
        y = np.expm1(y)
        y[y < 0.85] = 0
        res_site_week0 = smape(test2.Visits_true, y)
        # print(site, week, 'smape: %0.5f' % res_site_week0)
        
        if site not in ['commons.wikimedia.org', 'www.mediawiki.org',]:
        
            mask = smape_mask(test2.Visits_true, y, threshold)
            test3 = test2[mask]
            lr.fit(test3[num_cols], test3.Visits_norm)
            y = lr.predict(test2[num_cols])
            y += test2.AllVisits
            y = np.expm1(y)
            y[y < 0.85] = 0 # less than 1 is zero, why 0.85 for threshold?
            res_site_week = smape(test2.Visits_true, y)
            #print(site, week, 'smape: %0.5f' % res_site_week)
        else:
            res_site_week = res_site_week0
            
        print(site, week, 'smape: %0.5f' % res_site_week, 'delta: %0.5f' % (res_site_week0 - res_site_week))
        test.loc[(test.Site == site) & (test.Week == week), 'Visits'] = y
        
        
        res_site += res_site_week * test2.shape[0]
        res_site_den += test2.shape[0]
        res += res_site_week * test2.shape[0]
        res_den += test2.shape[0]
        out.append((test2.Visits_true.values, y, test2.shape[0]))
        
        test_all2 = test_all1[test_all1.Week == week]
        y = lr.predict(test_all2[num_cols])
        y += test_all2.AllVisits
        y = np.expm1(y)
        y[y < 0.85] = 0
        test_all.loc[(test_all.Site == site) & (test_all.Week == week), 'Visits'] = y

    res_site /= res_site_den
    print('smape %s: %0.5f' % (site, res_site))

        
res /= res_den
print('smape all: %0.5f' % res)

y_true = np.concatenate([y_true for (y_true, y_pred, size) in out], axis=0)
y_pred = np.concatenate([y_pred for (y_true, y_pred, size) in out], axis=0)
print('smape all: %0.5f' % smape(y_true, y_pred))

test.Visits = test.Visits.round(3)
test_all.Visits = test_all.Visits.round(3)

test[['Page', 'Date', 'Visits']].to_csv('/home/cdsw/data/submissions/pred_10_stage2_sept_10_train.csv', index=False)
test_all[['Id', 'Visits']].to_csv('/home/cdsw/data/submissions/pred_10_stage2_sept_10_test.csv', index=False)

test.head()