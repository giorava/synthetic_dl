import torch 
from torch import nn
import pandas as pd
import numpy as np
import pybedtools, tqdm, pyBigWig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class losses(nn.Module):
    def __init__(self, lam: float = 100):   
        super().__init__()
        self.l = lam ## weight of the mse given by 1/counts sum or something like that
    
    def multinomial_nll(self, task_target_counts, logits_task): 

        # task_target_counts should have shape (batch_size, output_length, num_tasks)
        # the logits_tracs should have shape (batch_size, output_length, num_tasks)

        
        pass 
        

        
    def forward(self, input, target):
        
        # for i in n_heads: 
        #       compute multinomial_nll on the profiles
        #       compute weighted MSE on the counts

        # sum all the losses
        # return the summed lossses
    


    # :param trueCounts: The experimentally-observed counts.
    #     Shape ``(batch-size x output-length x num-tasks)``
    # :param logits: The logits that the model is currently emitting.
    #     Shape ``(batch-size x output-length x num-tasks)``
    # :return: A scalar representing the profile loss of this batch.
    # """
    # logUtils.debug("Creating multinomial NLL.")
    # inputShape = ops.shape(trueCounts)
    # numBatches = inputShape[0]
    # numSamples = inputShape[1] * inputShape[2]  # output length * num_tasks

    # flatCounts = ops.reshape(trueCounts, [numBatches, numSamples])
    # flatLogits = ops.reshape(logits, [numBatches, numSamples])
    # totalCounts = ops.sum(flatCounts, axis=1)
    # distribution = tfp.distributions.Multinomial(total_count=totalCounts,
    #                                              logits=flatLogits)
    # logprobs = distribution.log_prob(flatCounts)
    # batchSize = ops.shape(trueCounts)[0]
    # sumProbs = ops.sum(logprobs)
    # curLoss = -sumProbs / ops.cast(batchSize, dtype=tf.float32)
    # return curLoss
