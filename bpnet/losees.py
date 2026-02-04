import torch 
from torch import nn
from torch.distributions import Multinomial

class BPNetLosses(nn.Module):
    
    def __init__(self, num_tasks: int):   
        super().__init__()

        self.num_tasks = num_tasks
        self.profile_loss = 0
        self.count_loss = 0

    def get_profile_loss(self): 
        return self.profile_loss

    def get_count_loss(self): 
        return self.count_loss

    def MSE(self, target_counts, pred_counts):
        
        tot_target_counts = torch.sum(target_counts, 1)
        tot_pred_counts = torch.sum(pred_counts, 1)
        mean_squared_error = torch.mean((tot_target_counts - tot_pred_counts)**2, 0)
        return mean_squared_error

    def multinomial_nll(self, target_profile, pred_logits): 

        batch_size = pred_logits.shape[0]

        flatTask = torch.flatten(target_profile, 1)
        flatCounts = torch.sum(flatTask, 1)
        
        flatLogits = torch.flatten(pred_logits, 1)
        
        batch_multinomial_nll = 0
        for i in range(batch_size): 
            
            dist = Multinomial(
                total_count=int(flatCounts[i].item()),
                logits=flatLogits[i]
            )
            mult_nll = -dist.log_prob(flatTask[i])
            batch_multinomial_nll += mult_nll
            
        return batch_multinomial_nll/batch_size
        
        
    def forward(self, pred_counts, target_counts, pred_prof, target_prof, count_weights):

        counts_raw_loss = self.MSE(
            target_counts = target_counts, 
            pred_counts = pred_counts
        )

        prof_raw_loss = self.multinomial_nll(
            target_profile = target_prof,
            pred_logits = pred_prof
        )

        self.profile_loss = prof_raw_loss
        self.count_loss = counts_raw_loss
        
        combined = prof_raw_loss+count_weights*counts_raw_loss

        return combined
    
