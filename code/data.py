import numpy as np

def generate_train_samples(train_data, batch_size, input_seq_len, output_seq_len):
  start_index = np.random.randint(0, train_data.shape[0] - output_seq_len - input_seq_len, batch_size)
  input_seqs = []
  output_seqs = []
  for start in start_index:
    input_seqs.append(train_data.iloc[start:start + input_seq_len].values)
    output_seqs.append(train_data.iloc[start + input_seq_len: start + input_seq_len + output_seq_len].values[:, 1])
  return np.concatenate(input_seqs).reshape((batch_size, input_seq_len, -1)), np.concatenate(output_seqs).reshape((batch_size, output_seq_len, -1))

def true_signal(x):
  y = 2 * np.sin(x)
  return y

def noise_func(x, noise_factor = 0.5):
  return np.random.randn(len(x)) * noise_factor

def generate_y_values(x):
  return true_signal(x) + noise_func(x)

def generate_noisy_samples(x, y_func, noise_func, batch_size, input_seq_len, output_seq_len): 
  
  total_start_points = len(x) - input_seq_len - output_seq_len
  
  # we have to have at least input_seq_len + output_seq_len points, so we
  # can start at any point in x that is at index total_start_points or less
  # here, choose, `batch_size` start indices. Two sequences will be generated
  # for each start index
  start_x_idx = np.random.choice(range(total_start_points), batch_size)

  input_seq_x = [x[i:i + input_seq_len] for i in start_x_idx]
  output_seq_x = [x[i + input_seq_len:i + input_seq_len + output_seq_len] for i in start_x_idx]

  input_seq_y = [y_func(x) + noise_func(x) for x in input_seq_x]
  output_seq_y = [y_func(x) + noise_func(x) for x in output_seq_x]

  ## return shape: (batch_size, time_steps, feature_dims)
  return np.array(input_seq_y), np.array(output_seq_y)