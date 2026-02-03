import numpy as np 
import sys, unittest, h5py, torch, os
sys.path.insert(0, "../src")
import data_managment_utils as data_utils


class test_data_utils(unittest.TestCase): 

    def __init__(self): 
        super().__init__

        self.X_array = np.array([[[1, 0, 0, 0], 
                            [0, 0, 0, 1], 
                            [0, 0, 1, 0], 
                            [0, 0, 1, 0], 
                            [0, 0, 0, 1]],
                            [[1, 0, 0, 0], 
                            [0, 0, 1, 0], 
                            [1, 0, 0, 0], 
                            [0, 0, 1, 0], 
                            [0, 0, 1, 0]  
                            ]], dtype = np.float16).transpose(0, 2, 1)
    
        self.Y_profile_pos = np.array([[1,2,3,4,5], 
                              [1,2,3,4,5]],  dtype = np.float16) 
        self.Y_profile_neg = -self.Y_profile_pos
        self.Y_profile = np.array([self.Y_profile_pos, self.Y_profile_neg])

        self.Y_counts_pos = self.Y_profile_pos.sum(1).reshape(-1, 1)
        self.Y_counts_neg = -self.Y_counts_pos
        self.Y_counts = np.array([self.Y_counts_pos, self.Y_counts_neg])

        data_utils.save_hdf5(
            X = self.X_array, y_counts= self.Y_counts, y_prof=self.Y_profile,
            filepath="test_data_manage", number_peaks=2, sequence_length=5, number_tasks=2
        )

    def test_h5f_saver(self):  

        h5 = h5py.File("test_data_manage", "r")
        assert list(h5.keys()) == ['data_X', 'data_y_counts_task_0', 'data_y_counts_task_1', 
                                   "data_y_prof_task_0", "data_y_prof_task_1"], "Keys are different than expected"
        assert np.all(h5["data_X"][1] == self.X_array[1, :, :]), "data_X entries are different than expected"
        assert np.all(h5["data_y_prof_task_0"][1] == self.Y_profile[0, :, :]), "data_y_prof entries are different than expected"
        assert np.all(h5["data_y_prof_task_1"][1] == self.Y_profile[1, :, :]), "data_y_prof entries are different than expected"
    
    def test_generator(self):

        dataset = data_utils.BPNetDataset(input_HDF5="test_data_manage", number_tasks = 2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        for batch in dataloader: 
            assert batch[0].shape == (1, 4, 5)
            assert batch[1].shape == (1, 5)
            assert batch[2].shape == (1, 1)
            assert batch[3].shape == (1, 5)
            assert batch[4].shape == (1, 1)

    def clean(self): 
        os.remove("test_data_manage")

if __name__=="__main__": 

    obj = test_data_utils()
    obj.test_h5f_saver()
    obj.test_generator()
    obj.clean()
