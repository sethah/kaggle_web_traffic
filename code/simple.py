import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%load_ext autoreload
%autoreload 2
%matplotlib inline

data_dir = "/home/cdsw/data/"
df = pd.read_csv(data_dir + "train_2.csv")

df = df.set_index('Page').T.rename_axis(None, axis=1).rename_axis('time')#.reset_index()

print("Dataframe has %d rows and %d columns" % df.shape)
df.index = df.index.to_datetime()

print(df.index.max().date(), df.index.min().date())

train_end = pd.to_datetime("2016-09-10")
test_start = pd.to_datetime("2016-09-13")
test_end = pd.to_datetime("2016-11-13")

train = df.loc[:train_end]
test = df.loc[test_start:test_end]

print("Train range: %s - %s" % (train.index.min().date(), train.index.max().date()))
print("Test range: %s - %s" % (test.index.min().date(), test.index.max().date()))

train.iloc[:, :3].head()

#page_data = [page.split("_") for page in train.columns]
# normalize data by log(1 + views)
train_norm = np.log1p(train)
train_norm.iloc[:, :3].head()
test.head()
pd.melt(test, value_vars=[test.columns])

# our task is to add a column to this dataframe with predictions
test_unrolled = pd.DataFrame(test.stack()).reset_index()
test_unrolled.columns = ['date', 'page', 'true_views']
test_unrolled.head()