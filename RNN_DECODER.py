import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN_DECODER(torch.nn.Module):

	def __init__(self, input_vector_size, context_vector_size, hidden_state_size, num_layers, output_size):

	  	self.lstm = nn.LSTM(input_size = input_vector_size + context_vector_size, hidden_size = hidden_state_size, num_layers = num_layers, batch_first = True, bidirectional = False) 
	  	self.FCN = nn.Sequential(
                nn.Linear(hidden_state_size, output_size, bias=False),
                #nn.ReLU(),
                #nn/Linear(2*hidden_state_size, output_size, bias = False)
	  		)


	def forward(self, context_vector, padded_input, lengths):

		_context_vector = torch.unsqueeze(context_vector, dim=1).expand(-1,padded_input.size()[1],-1)
		
		decoder_padded_input = torch.cat((_context_vector,padded_input), dim=-1)

		packed_input = pack_padded_sequence(decoder_padded_input, lengths, batch_first=True, enforce_sorted = False)

		packed_rnn_output, states = self.lstm(packed_input)

		padded_rnn_output, _lengths = pad_packed_sequence(packed_rnn_output, batch_first = True, padding_value=0, total_length= padded_input.size()[1]) 

		padded_output = self.FCN(padded_rnn_output)

		return padded_output