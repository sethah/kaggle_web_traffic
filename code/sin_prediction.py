import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from utils import persistence_forecast
import torch
from torch.autograd import Variable

from model import EncoderRNN, DecoderRNN

from data import generate_noisy_samples
%load_ext autoreload
%autoreload 2
%matplotlib inline

hidden_size = 16
input_size = 1  # dimensionality of each element in the sequence
output_size = 1
batch_size = 10
in_seq_len = 15
out_seq_len = 20
enc_layers = 1
dec_layers = 1

noise_func = lambda x: np.random.randn(len(x)) * 0.5
y_func = lambda x: 2.0 * np.sin(x)
t = np.linspace(0, 30, 105)
in_seq, out_seq = generate_noisy_samples(t, y_func, noise_func, 
                                         batch_size=batch_size, 
                                         input_seq_len=in_seq_len, 
                                         output_seq_len=out_seq_len)

in_seq.shape, out_seq.shape

plt.scatter(np.arange(in_seq.shape[1]), in_seq[0, :])

encoder = EncoderRNN(input_size, hidden_size, n_layers=enc_layers)
decoder = DecoderRNN(output_size, hidden_size, n_layers=dec_layers)

inp = Variable(torch.FloatTensor(in_seq).view(batch_size, in_seq_len, 1))
assert list(inp.size()) == [batch_size, in_seq_len, input_size]

out, hidden = encoder.forward(inp, encoder.initHidden(batch_size))
assert list(out.size()) == [batch_size, in_seq_len, hidden_size]
assert list(hidden.size()) == [enc_layers, batch_size, hidden_size]

SOS = 0.0
dec_inp = Variable(
    torch.FloatTensor([SOS] * (batch_size * 1 * output_size))\
    .view((batch_size, 1, output_size))
  )

dec_out, dec_hidden = decoder.forward(dec_inp, hidden)
assert list(dec_out.size()) == [batch_size, 1, output_size]
assert list(dec_hidden.size()) == [dec_layers, batch_size, hidden_size]

def req_grad_params(o):
  return (p for p in o.parameters() if p.requires_grad)

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

  target_length = out_seq.shape[1]
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
    in_seq, out_seq = generate_noisy_samples(t, y_func, noise_func, 
                                           batch_size=batch_size, 
                                           input_seq_len=in_seq_len, 
                                           output_seq_len=out_seq_len)
    inp = Variable(torch.FloatTensor(in_seq).view(batch_size, in_seq_len, 1))
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
train_epochs(1000, encoder, decoder, 0.0001)


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


eval_batches = 3
eval_steps = 50
in_seqs, out_seqs = generate_noisy_samples(t, y_func, noise_func, 
                                           batch_size=eval_batches, 
                                           input_seq_len=30, 
                                           output_seq_len=eval_steps)
in_seqs = in_seqs.reshape((eval_batches, 30, 1))
out_seqs = out_seqs.reshape((eval_batches, eval_steps, 1))
pred_seqs = evaluate(in_seqs, num_steps=eval_steps, output_size=1)

def p(in_seqs, out_seqs, pred_seqs):
  fig, axs = plt.subplots(1, pred_seqs.shape[0], figsize=(15, 3))
  for i, ax in enumerate(axs.reshape(-1)):
    ax.scatter(np.arange(in_seqs.shape[1]), in_seqs[i, :])
    ax.scatter(in_seqs.shape[1] + np.arange(out_seqs.shape[1]), out_seqs[i, :])
    ax.scatter(in_seqs.shape[1] + np.arange(pred_seqs.shape[1]), pred_seqs[i, : , 0])

p(in_seqs, out_seqs, pred_seqs)

