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
        """
        Compute Mean Squared Error (MSE) between total counts of target and predicted distributions.
        
        This loss function compares the summed counts across categories for each sample,
        suitable for regression tasks where total counts should match between target and prediction.
        
        Args:
            target_counts (torch.Tensor): Target count tensor of shape [batch_size, num_categories]
            pred_counts (torch.Tensor): Predicted count tensor of shape [batch_size, num_categories]
            
        Returns:
            torch.Tensor: Scalar MSE loss between total target and predicted counts
        """
        
        tot_target_counts = torch.sum(target_counts, 1)
        tot_pred_counts = torch.sum(pred_counts, 1)
        mean_squared_error = torch.mean((tot_target_counts - tot_pred_counts)**2, 0)
        return mean_squared_error

    def multinomial_nll(self, target_profile, pred_logits): 
        """
        Compute batched Multinomial Negative Log-Likelihood loss.
        
        Iterates over batch dimension to handle variable total counts per sample,
        computing NLL for each multinomial distribution defined by predicted logits
        against flattened target profile counts.
        
        Args:
            target_profile (torch.Tensor): Target profile counts, shape [batch_size, tasks, length]
            pred_logits (torch.Tensor): Predicted logits, shape [batch_size, tasks, length]
            
        Returns:
            torch.Tensor: Average NLL loss across batch (scalar)
        """

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
        """
        Combined loss computation for count matching and profile distribution modeling.
        
        Computes MSE loss on total counts and Multinomial NLL loss on flattened profiles,
        then combines them with a weighting factor for the count component.
        
        Args:
            pred_counts (torch.Tensor): Predicted total counts, shape [batch_size, num_categories]
            target_counts (torch.Tensor): Target total counts, shape [batch_size, num_categories]  
            pred_prof (torch.Tensor): Predicted profile logits, shape [batch_size, tasks, length]
            target_prof (torch.Tensor): Target profile counts, shape [batch_size, tasks, length]
            count_weights (float or torch.Tensor): Weighting factor for count loss contribution
            
        Returns:
            torch.Tensor: Combined scalar loss = profile_nll + count_weights * count_mse
        """

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
    
