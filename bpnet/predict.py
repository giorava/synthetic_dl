import logging
import sys, torch
import tqdm
from torch.utils.data import DataLoader
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
sys.path.insert(0, "../src")
import data_managment_utils as data_utils


class predictBPNet(): 
    
    def __init__(self, path_train_dataset, path_val_dataset, path_test_dataset, 
                 model, state_path, output_path,
                 number_tasks = 2, batch_size = 64): 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset_train = data_utils.BPNetDataset(input_HDF5=path_train_dataset, number_tasks = number_tasks, device = self.device)
        dataset_val = data_utils.BPNetDataset(input_HDF5=path_val_dataset, number_tasks = number_tasks, device = self.device)
        dataset_test = data_utils.BPNetDataset(input_HDF5=path_test_dataset, number_tasks = number_tasks, device = self.device)
        self.dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
        self.dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle = False)
        self.dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)

        state_dict = torch.load(state_path, map_location=self.device)
        # self.model = model
        # self.model = self.model.load_state_dict(state_dict)
        self.model = model          # your instantiated nn.Module
        self.model.load_state_dict(state_dict)  # <- no assignment here
        self.model.to(self.device)  # optional, if not already on device

        self.output_path = output_path

    def save_hdf5(self, predictions, fname):

        output_path = self.output_path+"/"+fname
        logging.info(f"Saving {output_path}")
        with h5py.File(output_path, "w") as f:
            f.create_dataset("profiles_logits", data = predictions[0].cpu().numpy())
            f.create_dataset("log10_counts", data = predictions[1].cpu().numpy())
            f.create_dataset("distributed_profiles_counts", data = predictions[2].cpu().numpy())

    def distribute_reads(self, profile_logits, log10_counts):

        total_counts = 10**log10_counts  # Convert log10 counts back to raw counts
        flatlogits = torch.flatten(profile_logits, 1)
        profile_probs = torch.softmax(flatlogits, dim=-1)  # Convert logits to probabilities
        distributed_counts = profile_probs * total_counts
        distributed_counts = distributed_counts.reshape(profile_logits.shape)  # Reshape back to original profile shape
        return distributed_counts 

    def evaluate_set(self, data_loader):
        self.model.eval()

        with torch.no_grad(): 
            for i, data in tqdm.tqdm(enumerate(data_loader)): 

                profile_pred, counts_pred = self.model(data[0])
                distributed_counts = self.distribute_reads(profile_pred, counts_pred)

                if i == 0: 
                    profile_preds = profile_pred
                    counts_preds = counts_pred
                    distributed_counts_preds = distributed_counts
                else: 
                    distributed_counts_preds = torch.cat([distributed_counts_preds, distributed_counts], 0)
                    profile_preds = torch.cat([profile_preds, profile_pred], 0)
                    counts_preds = torch.cat([counts_preds, counts_pred], 0)

        print(f"Profile predictions shape: {profile_preds.shape}")
        print(f"Counts predictions shape: {counts_preds.shape}")
        print(f"Distributed counts predictions shape: {distributed_counts_preds.shape}")
        return profile_preds, counts_preds, distributed_counts_preds
    
    def run_all_sets(self): 

        pred_train = self.evaluate_set(self.dataloader_train)
        self.save_hdf5(pred_train, "train_pred.h5")

        pred_val = self.evaluate_set(self.dataloader_val)
        self.save_hdf5(pred_val, "val_pred.h5")

        pred_test = self.evaluate_set(self.dataloader_test)
        self.save_hdf5(pred_test, "test_pred.h5")


