import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiRNN_ENCODER(torch.nn.Module):

    def __init__(self,input_vector_size,hidden_state_size,num_layers):

      	        super(BiRNN_ENCODER,self).__init__()
      	        self.bilstm = nn.LSTM(input_size = input_vector_size, hidden_size = hidden_state_size, num_layers = num_layers, batch_first = True, bidirectional = True)
      	        self.input_vector_size = input_vector_size
      	        self.hidden_state_size = hidden_state_size
      	        self.num_layers = num_layers


    def forward(self, padded_input, lengths):

                packed_input = pack_padded_sequence(padded_input, lengths, batch_first = True, enforce_sorted = False) 	           
                packed_output, states = self.bilstm(packed_input)
                padded_output, _lengths = pad_packed_sequence(packed_output, batch_first = True, padding_value = -1e320)
                h_n = states[0].view(self.num_layers,  2, -1, self.hidden_state_size)
                maxpoolv,ind = torch.max(input = padded_output,dim=1) #[batch_size,2*hidden_state_size]
                h_fb = torch.cat((h_n[-1,0],h_n[-1,1]),dim=1)
                context_vector = torch.cat((maxpoolv,h_fb),dim=1)
                return context_vector 