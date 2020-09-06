import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle

class CustomDataset(torch.nn.Module):

	def __init__(self, datafile):
	      self.data = pickle.load(open(datafile,'rb'))

    def __len__(self):
    	return len(self.data)

    def __getitem__(self,idx):
        return torch.tensor(self.data[idx])

def Custom_Collate(batch):
    
    lengths = torch.tensor([len(t) for t in batch])
    padded_batch = pad_sequence(batch, batch_first = True, padding_value = 0)
    return padded_batch, lengths 

def file_line_to_list(filename):
    
    mylist = []

    with open(filename,'r') as ufile:

         for fline in ufile:

            mylist.append(fline.rstrip('\n'))

    return mylist

def save_to_pickle(arr,filename):

	pickle.dump(arr,open(filename,'wb')) 
	
