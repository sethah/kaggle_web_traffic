import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from utils import persistence_forecast
import torch
from torch.autograd import Variable

from model import EncoderRNN, DecoderRNN

import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from data import generate_noisy_samples
%load_ext autoreload
%autoreload 2
%matplotlib inline

data_dir = "/home/cdsw/data/"
df = pd.read_csv(data_dir + "train_1.csv")

df = df.set_index('Page').T.rename_axis(None, axis=1).rename_axis('time')#.reset_index()

print("Dataframe has %d rows and %d columns" % df.shape)
df.index = df.index.to_datetime()

got_col = 'Game_of_Thrones_(season_1)_en.wikipedia.org_desktop_all-agents'
got = df[got_col]
got_mean = got.mean()
got_std = got.std()
got_std = (got - got_mean) / got_std
got_std.plot()

def in_season(dt):
  return dt > pd.to_datetime('2016-04-15') and dt < pd.to_datetime('2016-06-27')
indf = pd.DataFrame({'views': got_std.values, 'in_season': got_std.index.map(in_season).astype(float)}, index=got_std.index)
split_date = pd.to_datetime('2016-06-01')
train_data, test_data = indf[indf.index <= split_date], indf[indf.index > split_date]
def p(): train_data.plot(); test_data.plot()
p()
  
  
def generate_train_samples(train_data, batch_size, input_seq_len, output_seq_len):
  total_start_points = train_data.shape[0] - input_seq_len - output_seq_len
  
  # we have to have at least input_seq_len + output_seq_len points, so we
  # can start at any point in x that is at index total_start_points or less
  # here, choose, `batch_size` start indices. Two sequences will be generated
  # for each start index
  start_x_idx = np.random.choice(range(total_start_points), batch_size)

  input_seq_y = [train_data.iloc[i:i + input_seq_len].values for i in start_x_idx]
  output_seq_y = [train_data[['views']].iloc[i + input_seq_len:i + input_seq_len + output_seq_len].values for i in start_x_idx]

  ## return shape: (batch_size, time_steps, feature_dims)
  return np.array(input_seq_y), np.array(output_seq_y)

input_size = 2
output_size = 1
in_seq_len = 50
out_seq_len = 10
batch_size = 16
SOS = 0.
hidden_size = 32
enc_layers = 1
dec_layers = 1

in_seqs, out_seqs = generate_train_samples(train_data, batch_size, in_seq_len, out_seq_len)

encoder = EncoderRNN(input_size, hidden_size, n_layers=enc_layers)
decoder = DecoderRNN(output_size, hidden_size, n_layers=dec_layers)

inp = Variable(torch.FloatTensor(in_seqs).view(batch_size, in_seq_len, input_size))
assert list(inp.size()) == [batch_size, in_seq_len, input_size]

out, hidden = encoder.forward(inp, encoder.initHidden(batch_size))
assert list(out.size()) == [batch_size, in_seq_len, hidden_size]
assert list(hidden.size()) == [enc_layers, batch_size, hidden_size]

def train(inp, target, encoder, decoder, enc_opt, dec_opt, criterion, teacher_forcing_prob=0.5):
  '''
  Train the encoder, decoder on a single batch of input/output sequences.
  
  The input/output sequences can be of different lengths. Use teacher 
  forcing some percentage of the time.
  '''
  
  enc_opt.zero_grad()
  dec_opt.zero_grad()
  
  enc_hidden = encoder.initHidden(batch_size)
  enc_out, dec_hidden = encoder.forward(inp, enc_hidden)
  
  # start the decoder with an input of all SOS - start of sequence
  dec_inp = Variable(
    torch.FloatTensor([SOS] * (batch_size * 1 * output_size))\
    .view((batch_size, 1, output_size))
  )

  target_length = target.shape[1]
  loss = 0.0
  
  use_teacher_forcing = np.random.random() < teacher_forcing_prob
  
  if use_teacher_forcing:
    for di in range(target_length):
      # predict one step, add loss, and then use the true label as next input
      dec_out, dec_hidden = decoder.forward(dec_inp, dec_hidden)
      dec_inp = Variable(torch.FloatTensor(target[:, di]).contiguous().view((batch_size, 1, output_size)))
      # dec_out is the prediction, dec_inp is the actual label and will be used
      # as the next input
      loss += criterion(dec_out, dec_inp)
  else:
    for di in range(target_length):
      dec_out, dec_hidden = decoder.forward(dec_inp, dec_hidden)
      loss += criterion(dec_out, Variable(torch.FloatTensor(target[:, di]).contiguous().view(batch_size, 1, output_size)))
      dec_inp = dec_out

  loss.backward()  # compute the gradients
  enc_opt.step(); dec_opt.step()  # update parameters
  
  return loss.data[0] / target_length
  
def req_grad_params(o):
    return (p for p in o.parameters() if p.requires_grad)
  
def train_epochs(num_epochs, encoder, decoder, lr, print_every=100):
  '''
  Train encoder/decoder for a number of epochs. Each epoch, the train
  function is called and computes a backwards pass for a single batch
  of in/out sequences.
  
  The average loss per epoch is tracked and logged every print_every epochs.
  '''
  enc_opt = torch.optim.Adam(req_grad_params(encoder), lr=lr)
  dec_opt = torch.optim.Adam(req_grad_params(decoder), lr=lr)
  criterion = torch.nn.MSELoss()

  loss_total = 0.0

  for epoch in range(num_epochs):
    in_seq, out_seq = generate_train_samples(train_data,
                                             batch_size=batch_size, 
                                             input_seq_len=in_seq_len, 
                                             output_seq_len=out_seq_len)
    inp = Variable(torch.FloatTensor(in_seq).view(batch_size, in_seq_len, input_size))
    epoch_loss = train(inp, out_seq, encoder, decoder, enc_opt, dec_opt, criterion, teacher_forcing_prob=0.5)
    loss_total += epoch_loss
    if epoch % print_every == print_every - 1:
      print("[%d/%d] Avg. loss per epoch: %0.3f" % (epoch + 1, num_epochs, loss_total / print_every))
      loss_total = 0.0

encoder = EncoderRNN(input_size, hidden_size, n_layers=enc_layers)
for tensor in encoder.parameters():
  if len(list(tensor.size())) < 2:
    torch.nn.init.uniform(tensor)
  else:
    torch.nn.init.xavier_uniform(tensor)
decoder = DecoderRNN(output_size, hidden_size, n_layers=dec_layers)
for tensor in decoder.parameters():
  if len(list(tensor.size())) < 2:
    torch.nn.init.uniform(tensor)
  else:
    torch.nn.init.xavier_uniform(tensor)

train_epochs(1000, encoder, decoder, 0.005)

def evaluate(in_seqs, num_steps, output_size):
  '''
  Generate predictions for configurable batch_size and number 
  of steps to predict.
  '''

  inp = Variable(torch.FloatTensor(in_seqs))
  enc_hidden = encoder.initHidden(inp.size()[0])
  enc_out, dec_hidden = encoder.forward(inp, enc_hidden)

  # start the decoder with an input of all SOS - start of sequence
  dec_inp = Variable(
    torch.FloatTensor([SOS] * (inp.size()[0] * 1 * output_size))\
    .view((inp.size()[0], 1, output_size))
  )
  step_predictions = []
  for di in range(num_steps):
    dec_out, dec_hidden = decoder.forward(dec_inp, dec_hidden)
    step_predictions.append(dec_out.data.numpy())
    dec_inp = dec_out
  return np.concatenate(step_predictions, axis=1)

def predict_range(df, start, end, days):
  in_seqs = df.loc[start:end].values.reshape(1, -1, input_size)
  preds = evaluate(in_seqs, days, 1)
  preds = np.concatenate([np.ones(in_seqs.shape[1]) * np.nan, preds.ravel()])
  return pd.DataFrame({'true': df.loc[start:end + pd.Timedelta(days, unit='D')].views,
                'predicted': preds}, index=df.loc[start:end + pd.Timedelta(days, unit='D')].index)

r = predict_range(indf, pd.to_datetime('2016-06-01') - pd.Timedelta(50, 'D'), pd.to_datetime('2016-06-01'), 150)
r.plot()

eval_batches = 3
eval_steps = 50
in_seqs, out_seqs = generate_train_samples(train_data, 
                                           batch_size=eval_batches, 
                                           input_seq_len=50, 
                                           output_seq_len=eval_steps)
in_seqs = in_seqs.reshape((eval_batches, 50, input_size))
out_seqs = out_seqs.reshape((eval_batches, eval_steps, 1))
pred_seqs = evaluate(in_seqs, num_steps=eval_steps, output_size=1)

def p(in_seqs, out_seqs, pred_seqs):
  fig, axs = plt.subplots(1, pred_seqs.shape[0], figsize=(15, 3))
  for i, ax in enumerate(axs.reshape(-1)):
    ax.scatter(np.arange(in_seqs.shape[1]), in_seqs[i, :])
    ax.scatter(in_seqs.shape[1] + np.arange(out_seqs.shape[1]), out_seqs[i, :], label='true')
    ax.scatter(in_seqs.shape[1] + np.arange(pred_seqs.shape[1]), pred_seqs[i, : , 0])
    ax.legend()

p(in_seqs[:, :, 1], out_seqs, pred_seqs)

#smt.seasonal_decompose(train_data.views).plot();
#
#mod = smt.SARIMAX(train_data.views, trend='c', order=(8, 1, 1))
#res = mod.fit()
#
#pred_dy = res.get_prediction(start='2015-10-01', dynamic='2015-10-01')
#pred_dy_ci = pred_dy.conf_int()
#
#def p1():s
#
##  ax = train_data.views.plot(label='observed')
#  plt.plot(train_data.index, train_data.views)
#  plt.plot(pred_dy.predicted_mean.index, pred_dy.predicted_mean)
##  ax.fill_between(pred_dy_ci.index,
##                  pred_dy_ci.iloc[:, 0],
##                  pred_dy_ci.iloc[:, 1], color='k', alpha=.25)
##  ax.set_ylabel("Monthly Flights")
#  plt.plot(train_data.loc[start:end].index, pred_seqs[0, :, 0])
#
#  # Highlight the forecast area
##  ax.fill_betweenx(ax.get_ylim(), pd.Timestamp('2013-01-01'), train_data.views.index[-1],
##                   alpha=.1, zorder=-1)
##  ax.annotate('Dynamic $\\longrightarrow$', (pd.Timestamp('2013-02-01'), 550))
#
##  plt.legend()
##  sns.despine()  
#p1()
#
#in_seqs = train_data.loc['2015-07-01':'2015-10-01'].values.reshape((1, -1, 2))
#pred_seqs = evaluate(in_seqs, 180, output_size=1)
#
#start = pd.to_datetime('2015-10-01')
#end = pd.to_datetime('2015-10-01') + pd.Timedelta(179, unit='D')
#
#plt.plot(train_data.loc[start:end].index, pred_seqs[0, :, 0])
#plt.plot(train_data.loc[start:end].index, train_data.loc[start:end]['views'])


