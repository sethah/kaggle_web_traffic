import torch
from torch.autograd import Variable

class EncoderRNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, dropout_p=0.5, n_layers=2):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.gru = torch.nn.GRU(input_size, hidden_size, num_layers=n_layers, dropout=dropout_p,
                      batch_first=True)

  def forward(self, inp, hidden):
    return self.gru(inp, hidden)

  def initHidden(self, batch_size):
    return torch.autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
      
class DecoderRNN(torch.nn.Module):
  def __init__(self, output_size, hidden_size, dropout_p=0.5, n_layers=2):
    super(DecoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.gru = torch.nn.GRU(output_size, hidden_size, num_layers=n_layers, dropout=dropout_p, 
                            batch_first=True)
    self.out = torch.nn.Linear(hidden_size, output_size)
    self.dropout_p = dropout_p
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, inp, hidden):
    res, hidden = self.gru(inp, hidden)
    res = self.dropout(res)
    return self.out(res), hidden