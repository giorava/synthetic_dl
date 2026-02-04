import torch 
from torch import nn
import pandas as pd
import numpy as np
import pybedtools, tqdm, pyBigWig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BPNetSingleHead(nn.Module): 
    def __init__(self, numFilters: int, widthFilters: int, n_convolutions: int):
        super().__init__()
        self.numFilters = numFilters
        self.widthFilters = widthFilters
        self.n_convolutions = n_convolutions - 1

        # Fixed layers created once
        self.initial_conv = nn.Conv1d(4, numFilters, kernel_size=widthFilters, padding=0)
        self.relu = nn.ReLU()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(numFilters, numFilters, kernel_size=3, dilation=2**(i+1), padding=0)
            for i in range(self.n_convolutions)
        ])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(numFilters, 1)

    def forward(self, x):
        out = self.relu(self.initial_conv(x))

        for i, conv in enumerate(self.dilated_convs):
            new_out = self.relu(conv(out))
            crop_of = (out.shape[-1] - new_out.shape[-1]) // 2
            previous_output = out[:, :, crop_of:-crop_of]
            out = previous_output + new_out

        out = self.adaptive_pool(out).squeeze(-1)
        out = self.fc(out)
        return out    
    

class BPNetSingleHeadProfile(nn.Module):
    def __init__(self, numFilters: int, widthFilters: int, n_convolutions: int, number_tasks: int):
        
        super().__init__()
        self.numFilters = numFilters
        self.widthFilters = widthFilters
        self.n_convolutions = n_convolutions - 1

        self.initial_conv = nn.Conv1d(4, numFilters, kernel_size=widthFilters, padding=0)
        self.relu = nn.ReLU()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(numFilters, numFilters, kernel_size=3, dilation=2**(i+1), padding=0)
            for i in range(self.n_convolutions)
        ])

        self.last_conv = nn.Conv1d(numFilters, number_tasks, kernel_size = widthFilters, padding = 0)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(numFilters, number_tasks)

    def forward(self, x): 
        out = self.relu(self.initial_conv(x))

        # binding core model
        for i, conv in enumerate(self.dilated_convs):
            new_out = self.relu(conv(out))
            crop_of = (out.shape[-1] - new_out.shape[-1]) // 2
            previous_output = out[:, :, crop_of:-crop_of]
            out = previous_output + new_out

        # specific profile heads
        profile_head = self.last_conv(out)
        count_head = self.adaptive_pool(out).squeeze(-1)
        count_head = self.fc(count_head)
        return profile_head, count_head


class BPNetSingleHeadFc(nn.Module): 
    
    def __init__(self, numFilters: int, widthFilters: int, n_convolutions: int): 
        super().__init__()
        self.numFilters = numFilters
        self.widthFilters = widthFilters
        self.n_convolutions = n_convolutions - 1

        # Fixed layers created once
        self.initial_conv = nn.Conv1d(4, numFilters, kernel_size=widthFilters, padding=0)
        self.relu = nn.ReLU()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(numFilters, numFilters, kernel_size=3, dilation=2**(i+1), padding=0)
            for i in range(self.n_convolutions)
        ])
        
        ## add an additional convolution without relu and a fully connected layer to get the counts
        self.last_conv = nn.Conv1d(numFilters, 1, kernel_size = widthFilters, padding = 0)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1, 1)

    def forward(self, x): 
        out = self.relu(self.initial_conv(x))

        for i, conv in enumerate(self.dilated_convs):
            new_out = self.relu(conv(out))
            crop_of = (out.shape[-1] - new_out.shape[-1]) // 2
            previous_output = out[:, :, crop_of:-crop_of]
            out = previous_output + new_out

        out = self.last_conv(out)
        out = self.adaptive_pool(out).squeeze(-1)
        out = self.fc(out)
        return out
            
class BPNetMultiHeadProfile(nn.Module):

    def __init__(self, numFilters: int, widthFilters: int, n_convolutions: int,
                 n_heads: int, n_tasks: int):
        super().__init__()
        self.numFilters = numFilters
        self.widthFilters = widthFilters
        self.n_convolutions = n_convolutions - 1
        self.n_heads = n_heads
        self.n_tasks = n_tasks

        # Fixed layers created once
        self.initial_conv = nn.Conv1d(4, numFilters, kernel_size=widthFilters, padding=0)
        self.relu = nn.ReLU()
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(numFilters, numFilters, kernel_size=3, dilation=2**(i+1), padding=0)
            for i in range(self.n_convolutions)
        ])

        # fully connected counts head
        self.counts_head = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(numFilters, 1)
            )
            for i in range(self.n_heads)
        ])

        # heads for the profile counts
        self.profile_head = nn.ModuleList([
            nn.Conv1d(numFilters, self.n_tasks, kernel_size = widthFilters, padding = 0)
            for i in range(self.n_heads)
        ])


    def forward(self, x):
        
        out = self.relu(self.initial_conv(x))

        for i, conv in enumerate(self.dilated_convs):
            new_out = self.relu(conv(out))
            crop_of = (out.shape[-1] - new_out.shape[-1]) // 2
            previous_output = out[:, :, crop_of:-crop_of]
            out = previous_output + new_out

        counts = []
        profiles = []
        for i in range(self.n_heads):
            counts.append(self.counts_head[i](out))
            profiles.append(self.profile_head[i](out))

        return counts, profiles  

