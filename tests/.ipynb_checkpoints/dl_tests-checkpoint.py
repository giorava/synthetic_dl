import sys, unittest, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "../src")
import data_managment_utils as data_utils

sys.path.insert(0, "../bpnet")
import models, losses


class test_dl(unittest.TestCase): 

    def __init__(self, output_size, w_filters, n_dil_convolution): 

        super().__init__


        self.output_size = output_size
        self.w_filters = w_filters
        self.n_dil_convolutions = n_dil_convolution

        self.dataset = data_utils.BPNetDataset("../demos/data_dl/train.h5", 2)
        self.dataloader = DataLoader(dataset = self.dataset, batch_size = 64, shuffle = True)

        sample = next(iter(self.dataloader))
        print("Input shape", sample[0].shape)
        print("Exp profile shape", sample[1].shape)
        print("Exp counts shape", sample[2].shape)

        # self.input_size = input_utils.getInputLength(
        #     outPredLen = self.output_size, 
        #     numDilLayers = self.n_dil_convolutions, 
        #     initialConvolutionWidths = [self.w_filters],       
        #     verbose = True)

    def test_forward_pass(self): 

        input, profiles, counts = next(iter(self.dataloader))

        model = models.BPNetSingleHeadProfile(
            numFilters = 80, 
            widthFilters = self.w_filters, 
            n_convolutions = self.n_dil_convolutions+1, 
            number_tasks = 2
        )
        profile_pred, counts_pred = model.forward(input)

        assert profile_pred.shape == profiles.shape
        assert counts_pred.shape == counts.shape


    def test_losses(self): 

        input, profiles, counts = next(iter(self.dataloader))

        model = models.BPNetSingleHeadProfile(
            numFilters = 80, 
            widthFilters = self.w_filters, 
            n_convolutions = self.n_dil_convolutions+1, 
            number_tasks = 2
        )
        profile_pred, counts_pred = model.forward(input)
        loss = losses.BPNetLosses(num_tasks=2)
        total_loss = loss.forward(
            pred_counts = counts_pred, 
            target_counts = counts, 
            pred_prof = profile_pred, 
            target_prof = profiles, 
            count_weights = 1
        )
        print(total_loss)

    def test_training_loop(self): 

        model = models.BPNetSingleHeadProfile(
            numFilters = 80, 
            widthFilters = self.w_filters, 
            n_convolutions = self.n_dil_convolutions+1, 
            number_tasks = 2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_obj = losses.BPNetLosses(num_tasks=2)

        ## go over one loop
        input, profiles, counts = next(iter(self.dataloader))
        optimizer.zero_grad()
        profile_pred, counts_pred = model(input)
        loss = loss_obj.forward(
            pred_counts = counts_pred, 
            target_counts = counts, 
            pred_prof = profile_pred, 
            target_prof = profiles, 
            count_weights = 1
        )
        loss.backward()
        optimizer.step()

        print(loss_obj.get_profile_loss())
        print(loss_obj.get_count_loss())

if __name__=="__main__": 

    obj = test_dl(
        output_size = 1000,
        w_filters = 7, 
        n_dil_convolution = 9
    )

    obj.test_forward_pass()
    obj.test_losses()
    obj.test_training_loop()
