import torch 
from torch import nn
import pandas as pd
import numpy as np
import pybedtools, tqdm, pyBigWig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BPNetLosses(nn.Module):
    def __init__(self):   
        super().__init__()

        self.profile_loss = 0
        self.count_loss = 0

    def get_profile_loss(self): 
        return self.profile_loss

    def get_count_loss(self): 
        return self.count_loss

    def MSE(self, target_counts, pred_counts):

        # compute simple MSE, target counts tensor is BatchSize x 1 and so is the pred_counts vector

        pass

    
    def multinomial_nll(self, task_target_counts, logits_task): 

        # task_target_counts should have shape (batch_size, output_length)
        # the logits_tracs should have shape (batch_size, output_length)

        pass 
        
    def forward(self, input, target, counts_weights):
        
        # counts_loss = []
        # profile_los = []
        # for i in n_heads: 
        #       
        #       compute multinomial_nll on the profiles
        #       compute weighted MSE on the counts

        ## combined = torch.sum(counts_weight*counts_loss) + torch.sum(profile_los)
        ## return combined 

        # self.profile_loss = torch.sum(profile_los)
        # self.count_loss = torch.sum(counts_loss)
        pass 

