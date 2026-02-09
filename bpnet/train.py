import numpy as np 
import sys, unittest, h5py, torch, os
from torch.utils.data import DataLoader

sys.path.insert(0, "../src")
import data_managment_utils as data_utils
import input_utils

sys.path.insert(0, "../bpnet")
import models, losses
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import tqdm



class trainBPNet(): 

    def __init__(self): 

        self.optimizer = 1 # adam optimizer
        self.model = 2 # model definition
        self.dataloader_train
        self.dataloader_val
        self.dataloader_test
        self.initial_lambda_counts
        self.output_path
        self.fraction_profile = 0.1
        self.loss_obj = losses.BPNetLosses(num_tasks=2)


    def train_one_epoch(self, epoch_index, w_counts): 

        losses = []

        for i, data  in tqdm.tqdm(enumerate(self.dataloader_train)): 
            input, profiles, counts = data

            self.optimizer.zero_grad()                      # Zero your gradients for every batch!
            profile_pred, counts_pred = self.model(input)   # predict outputs

            # compute loss and gradient
            loss = self.loss_obj(
                pred_counts = counts_pred, 
                target_counts = counts, 
                pred_prof = profile_pred, 
                target_prof = profiles, 
                count_weights = w_counts
            )
            loss.backward()
            self.optimizer.step()            # update weights
            running_loss += [loss.item()]

        return np.mean(losses)
    
    def compute_lambda(self, lambda_e): 

        p, c, f = self.loss_obj.profile_loss(), self.loss_obj.count_loss(), self.fraction_profile
        
        lambda_ep1_prime = (p*f)/(1-f)*c
        lambda_ep1_second = 0.3*lambda_ep1_prime + 0.7*lambda_e
        gamma = lambda_ep1_second/lambda_e

        if gamma>2: 
            return 2*lambda_e
        if gamma<(1/2): 
            return lambda_e/2
        else: 
            return lambda_ep1_second


    def train(self, epochs): 

        lambda_counts = self.initial_lambda_counts
        best_vloss = 1_000_000
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        for epoch in range(epochs): 
            print('EPOCH {}:'.format(epoch_number + 1))
            current_lambda = lambda_counts

            self.model.train(True)
            average_loss = self.train_one_epoch(epoch_index = epoch, 
                                                w_counts = current_lambda)
            
            ## computing the validation loss
            validation_loss = []
            self.model.eval()
            with torch.no_grad(): 
                for i, vdata in enumerate(self.dataloader_val): 
                    input, profiles, counts = vdata
                    profile_pred, counts_pred = self.model(input)
                    vlos = self.loss_obj(
                        pred_counts = counts_pred, 
                        target_counts = counts, 
                        pred_prof = profile_pred, 
                        target_prof = profiles, 
                        count_weights = current_lambda
                    )
                    validation_loss.append(vlos)
            average_vlos = np.mean(validation_loss)

            ### updating the count weight
            lambda_counts = self.compute_lambda(lambda_e = current_lambda)


            writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : average_loss, 'Validation' : average_vlos},
                    epoch_number + 1)
            writer.flush()

            ## track performance and save model
            if average_vlos < best_vloss:
                best_vloss = average_vlos
                model_path = f'{self.output_path}/model_{timestamp}_{epoch_number}'
                torch.save(self.model.state_dict(), model_path)




