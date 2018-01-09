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

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Dense, Activation, Dropout, Flatten

from keras import regularizers 

import keras

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold

def init():
    np.random.seed = 0
    
init()

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def smape2D(y_true, y_pred):
    return smape(np.ravel(y_true), np.ravel(y_pred))
    
def smape_mask(y_true, y_pred, threshold):
    denominator = (np.abs(y_true) + np.abs(y_pred)) 
    diff = np.abs(y_true - y_pred) 
    diff[denominator == 0] = 0.0
    
    return diff <= (threshold / 2.0) * denominator
  
max_size = 181 # number of days in 2015 with 3 days before end

offset = 1 / 2

data_dir = "/home/cdsw/data/"
train_all = pd.read_csv(data_dir + "train_2.csv")
train_all.head()


all_page = train_all.Page.copy()
train_key = train_all[['Page']].copy()
train_all = train_all.iloc[:,1:] * offset 
train_all.head()


def get_date_index(date, train_all=train_all):
    for idx, c in enumerate(train_all.columns):
        if date == c:
            break
    if idx == len(train_all.columns):
        return None
    return idx
  
get_date_index('2016-09-13')
train_all.shape[1] - get_date_index('2016-09-10')
get_date_index('2017-09-10') - get_date_index('2016-09-10')


trains = []
tests = []
train_end = get_date_index('2016-09-10') + 1
test_start = get_date_index('2016-09-13')

for i in range(-3,4):
    train = train_all.iloc[ : , (train_end - max_size + i) : train_end + i].copy().astype('float32')
    test = train_all.iloc[:, test_start + i : 63 + test_start + i].copy().astype('float32')
    train = train.iloc[:,::-1].copy().astype('float32')
    trains.append(train)
    tests.append(test)

train_all = train_all.iloc[:,-(max_size):].astype('float32')
train_all = train_all.iloc[:,::-1].copy().astype('float32')

test_3_date = tests[3].columns

data = [page.split('_') for page in tqdm(train_key.Page)]

access = ['_'.join(page[-2:]) for page in data]

site = [page[-3] for page in data]

page = ['_'.join(page[:-3]) for page in data]
page[:2]

train_key['PageTitle'] = page
train_key['Site'] = site
train_key['AccessAgent'] = access
train_key.head()


train_norms = [np.log1p(train).astype('float32') for train in trains]
train_norms[3].head()

train_all_norm = np.log1p(train_all).astype('float32')
train_all_norm.head()

for i,test in enumerate(tests):
    first_day = i-2 # 2016-09-13 is a Tuesday
    test_columns_date = list(test.columns)
    test_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
    test.columns = test_columns_code

tests[3].head()


for test in tests:
    test.fillna(0, inplace=True)

    test['Page'] = all_page
    test.sort_values(by='Page', inplace=True)
    test.reset_index(drop=True, inplace=True)
    

tests = [test.merge(train_key, how='left', on='Page', copy=False) for test in tests]

tests[3].head()


test_all_id = pd.read_csv(data_dir + 'key_2.csv')

test_all_id['Date'] = [page[-10:] for page in tqdm(test_all_id.Page)]
test_all_id['Page'] = [page[:-11] for page in tqdm(test_all_id.Page)]
test_all_id.head()

test_all = test_all_id.drop('Id', axis=1)
test_all['Visits_true'] = np.NaN

test_all.Visits_true = test_all.Visits_true * offset
test_all = test_all.pivot(index='Page', columns='Date', values='Visits_true').astype('float32').reset_index()

test_all['2017-11-14'] = np.NaN
test_all.sort_values(by='Page', inplace=True)
test_all.reset_index(drop=True, inplace=True)

test_all.head()

test_all_columns_date = list(test_all.columns[1:])
first_day = 2 # 2017-13-09 is a Wednesday
test_all_columns_code = ['w%d_d%d' % (i // 7, (first_day + i) % 7) for i in range(63)]
cols = ['Page']
cols.extend(test_all_columns_code)
test_all.columns = cols
test_all.head()

test_all = test_all.merge(train_key, how='left', on='Page')
test_all.head()

for test in tests:
    test.reset_index(inplace=True)
test_all = test_all.reset_index()

test = pd.concat(tests[2:5], axis=0).reset_index(drop=True)
test.shape
test.head()