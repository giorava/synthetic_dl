import logging
import sys, torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import callbacks
sys.path.insert(0, "../src")
import data_managment_utils as data_utils
sys.path.insert(0, "../bpnet")
import losses


class trainBPNet(): 

    def __init__(self, model, optimizer, 
                 path_train_dataset: str, path_val_dataset: str, model_ouput: str, 
                 fraction_profile: float = 0.1, number_tasks: int = 2,
                 init_lambda: float = 32.0, batch_size: int = 64): 

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model and optimizer objects
        self.optimizer = optimizer
        self.model = model

        # hyperparams
        self.initial_lambda_counts = init_lambda
        self.model_ouput = model_ouput
        self.fraction_profile = fraction_profile
        self.loss_obj = losses.BPNetLosses(num_tasks=number_tasks)

        ## data loaders: 
        dataset_train = data_utils.BPNetDataset(input_HDF5=path_train_dataset, number_tasks = number_tasks, device = self.device)
        dataset_val = data_utils.BPNetDataset(input_HDF5=path_val_dataset, number_tasks = number_tasks, device = self.device)
        self.dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        self.dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False)

        self.batch_size = batch_size


    def train_one_epoch(self, w_counts): 

        losses = []

        for i, data  in enumerate(self.dataloader_train): 
            input, profiles, counts = data

            self.optimizer.zero_grad()                      # Zero your gradients for every batch
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
            self.optimizer.step()                          # update weights

            average_loss = loss.item()/input.shape[0]
            losses += [average_loss]
            print(f">>>> Processed batch {i} with average loss {average_loss} ", flush = True)

        ## smoothing the loss by the last batches 
        last_losses = np.array(losses)[-int(np.ceil(0.1*self.batch_size)):]
        return np.median(last_losses), losses
    

    def validate(self, current_lambda):

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
                validation_loss.append(vlos.item()/input.shape[0])

        ## smoothing the loss by the last batches
        last_vlosses = np.array(validation_loss)[-int(np.ceil(0.1*self.batch_size)):]
        return np.median(last_vlosses), validation_loss

    def train(self, epochs): 

        lambda_counts = self.initial_lambda_counts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0

        # scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        # early stopping
        early_stopping = callbacks.EarlyStopping(tolerance=15, min_delta=0)

        losses = []
        vlosses = []

        for _ in range(epochs): 
            print('EPOCH {}:'.format(epoch_number + 1))
            current_lambda = lambda_counts

            self.model.train(True)
            average_loss, _losses = self.train_one_epoch(w_counts = current_lambda)
            average_vlos, _vlosses = self.validate(current_lambda)
            scheduler.step(average_vlos)

            ### updating the count weight
            #lambda_counts = self.compute_lambda(lambda_e = current_lambda)
            lambda_counts == callbacks.compute_lambda(
                p = self.loss_obj.get_profile_loss(),
                c = self.loss_obj.get_count_loss(), 
                f = self.fraction_profile, 
                lambda_e=current_lambda
            )

            # save model parameters at each epoch
            logging.info('Epoch {}: Training Loss = {}, Validation Loss = {}, Lambda Counts = {}'.format(epoch_number + 1, average_loss, average_vlos, lambda_counts))
            model_path = f'{self.model_ouput}/model_{timestamp}_{epoch_number}'
            torch.save(self.model.state_dict(), model_path)
            losses.append(_losses)
            vlosses.append(_vlosses)

            early_stopping.criteria(train_loss=average_loss, validation_loss=average_vlos)
            if early_stopping.get_early_stop():
                logging.info(f"Early stopping at epoch: {epoch_number + 1}")
                break

            epoch_number += 1

        np.savetxt(f'{self.model_ouput}/batch_averaged_training_losses.txt', np.array(losses))
        np.savetxt(f'{self.model_ouput}/batch_averaged_validation_losses.txt', np.array(vlosses))
