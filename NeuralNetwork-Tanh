# Tanh-Ann-
import numpy as np

def tanh(x):
    return np.tanh(x)

def forward_pass(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = tanh(output_input)
    return output

inputs = np.array([0.6, 0.8])
np.random.seed(42)
weights_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))
weights_hidden_output = np.random.uniform(-0.5, 0.5, (2, 1))
bias_hidden = np.array([0.5, 0.7])
bias_output = 0.7

output = forward_pass(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
print(output)

output:
[0.23917015]
