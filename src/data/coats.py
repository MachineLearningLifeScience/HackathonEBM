
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

class COATS(Dataset):
    def __init__(self, root, download=True, which="mnar_train", prop_debias=0.6, prop_val=0.2, prop_test=0.2):
        """
        Download and prepare the COATS dataset for MNAR simulation.
        data: numpy array or torch tensor, shape (n_samples, n_features)
        download: whether to download the dataset from Kaggle
                /!\ In order to download from Kaggle, you need to have the Kaggle API installed and configured.
                Basically, it means having a kaggle.json file in your ~/.kaggle/ folder with username and key.
                    > Get your Kaggle API credentials
                    > Go to your Kaggle account: https://www.kaggle.com/settings
                    > Under API, click Create New API Token
                    It will download a file called kaggle.json — this contains your username and API key.
                    > Remember to make sure your kaggle.json file is in the correct location:
                    ~/.kaggle/kaggle.json
                    > Remember to set the correct permissions:
                    chmod 600 ~/.kaggle/kaggle.json
        which: one of ['mnar_train', 'mcar_train', 'val', 'test']
        prop_debias: proportion of mcar data to use for debiasing
        prop_val: proportion of mcar data to use for validation
        prop_test: proportion of mcar data to use for testing
        """
        self.root = root
        if download:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('kkkongxin/dataset-coat', path=self.root, unzip=True)
        
        self.check_folder_consistency()
        self.setup_data()
        self.prop_debias = prop_debias
        self.prop_val = prop_val
        self.prop_test = prop_test
        assert self.prop_debias + self.prop_val + self.prop_test == 1.0, "Proportions must sum to 1.0"
        assert which in ["mnar_train", "mcar_train", "mnar_val","mcar_val", "test"], "which must be one of ['mnar_train', 'mcar_train', 'mnar_val', 'mcar_val', 'test']"
        self.which = which
        self.matrix_created=False
        self.create_data_matrix()

    def check_folder_consistency(self):
        # Implement any necessary checks for folder consistency
        assert os.path.exists(self.root), "Dataset root folder does not exist at {}.".format(self.root)
        assert os.path.exists(os.path.join(self.root, 'coat', 'train.csv')), "MNAR data file is missing at {}.".format(os.path.join(self.root, 'coat', 'train.csv'))
        assert os.path.exists(os.path.join(self.root, 'coat', 'random.csv')), "MCAR data file is missing at {}.".format(os.path.join(self.root, 'coat', 'random.csv'))




    def setup_data(self):
        path_train = os.path.join(self.root, 'coat', 'train.csv')
        path_test = os.path.join(self.root, 'coat', 'random.csv')
        self.train_df = pd.read_csv(path_train, ) # No header in the random data
        self.test_df = pd.read_csv(path_test, ) # Header in the sampling data

        # Check the size of the train data
        # assert self.train_df.shape == (311704,3), "Train data shape is incorrect."
        # assert self.test_df.shape == (54000,3), "Test data shape is incorrect with shape {}.".format(self.test_df.shape)


    def create_data_matrix(self):
        # Create user-item matrix for train and test data
        n_users_train = self.train_df['user_id'].nunique()
        n_users_test = self.test_df['user_id'].nunique()
        print("We found {} users in the training set.".format(n_users_train))
        print("We found {} users in the test set.".format(n_users_test))
        n_items = np.union1d(self.train_df['item_id'].unique(), self.test_df['item_id'].unique()).shape[0]
        print("We found {} items in the training set.".format(n_items))
        assert n_items == 300, "Number of items is incorrect."


        mnar_matrix = np.zeros((n_users_train, n_items))
        mnar_mask_matrix = np.zeros((n_users_train, n_items))
        for row in self.train_df.itertuples():
            # mnar_matrix[row.user_id, row.item_id] = max(row.rating -1, 0)
            mnar_matrix[row.user_id, row.item_id] = row.rating-1
            mnar_mask_matrix[row.user_id, row.item_id] = 1.0 if row.rating > 0 else 0.0
        self.mnar_matrix = mnar_matrix
        self.mnar_mask_matrix = mnar_mask_matrix


        mcar_matrix = np.zeros((n_users_test, n_items))
        mcar_mask_matrix = np.zeros((n_users_test, n_items))
        for row in self.test_df.itertuples():
            mcar_matrix[row.user_id, row.item_id] = row.rating-1
            mcar_mask_matrix[row.user_id, row.item_id] = 1.0 if row.rating > 0 else 0.0
        self.mcar_matrix = mcar_matrix
        self.mcar_mask_matrix = mcar_mask_matrix


        indice_debias, indice_val = self.split_mcar_data(mcar_matrix)
        self.mcar_train = mcar_matrix[:indice_debias]
        self.mcar_val = mcar_matrix[indice_debias:indice_val]
        self.mcar_test = mcar_matrix[indice_val:]
        self.mcar_mask_train = mcar_mask_matrix[:indice_debias]
        self.mcar_mask_val = mcar_mask_matrix[indice_debias:indice_val]
        self.mcar_mask_test = mcar_mask_matrix[indice_val:]

        
        index_train = self.split_mnar_data(mnar_matrix)
        self.mnar_train = mnar_matrix[:index_train]
        self.mnar_mask_train = mnar_mask_matrix[:index_train]
        self.mnar_val = mnar_matrix[:index_train]
        self.mnar_mask_val = mnar_mask_matrix[:index_train]

        self.matrix_created = True
        
    def split_mcar_data(self, mcar_matrix):
        index_max = mcar_matrix.shape[0]
        indice_debias = int(index_max * self.prop_debias)
        indice_val = int(index_max * (self.prop_debias + self.prop_val))
        return indice_debias, indice_val
    
    def split_mnar_data(self, mnar_matrix):
        index_max = mnar_matrix.shape[0]
        index_train = int(index_max * (1.0 - self.prop_val))
        return index_train
    

    def __len__(self):
        if not self.matrix_created:
            self.create_data_matrix()


        if self.which == "mnar_train":
            return self.mnar_train.shape[0]
        elif self.which == "mcar_train":
            return self.mcar_train.shape[0]
        elif self.which == "mnar_val":
            return self.mnar_val.shape[0]
        elif self.which == "mcar_val":
            return self.mcar_val.shape[0]
        elif self.which == "test":
            return self.mcar_test.shape[0]

    def __getitem__(self, idx):
        if not self.matrix_created:
            self.create_data_matrix()

        if self.which == "mnar_train":
            x = self.mnar_train[idx]
            mask = self.mnar_mask_train[idx]
        elif self.which == "mcar_train":
            x = self.mcar_train[idx]
            mask = self.mcar_mask_train[idx]
        elif self.which == "mcar_val":
            x = self.mcar_val[idx]
            mask = self.mcar_mask_val[idx]
        elif self.which == "mnar_val":
            x = self.mnar_val[idx]
            mask = self.mnar_mask_val[idx]
        elif self.which == "test":
            x = self.mcar_test[idx]
            mask = self.mcar_mask_test[idx]

        x = torch.from_numpy(x).to(torch.float32)
        # assert torch.all(x<=4.0) and torch.all(x>=0.0), "Data values should be in [0, 4]"
        mask = torch.from_numpy(mask).to(torch.float32)
        assert torch.any(mask!=0), "At least one entry in the mask should be 1"
        return x, mask
    
if __name__ == "__main__":
    dataset = COATS(root="/scratch/hhjs/data/COATS/", download=True, which="mnar_train")
    print("Dataset length:", len(dataset))
    x, mask = dataset[0]
    print("First sample data shape:", x.shape)
    print("First sample mask shape:", mask.shape)
    #Count the number of missing entries in the mnar dataset
    num_missing = np.prod(dataset.mnar_mask_matrix.shape) - np.sum(dataset.mnar_mask_matrix)
    print("Number of missing entries in the mnar dataset:", num_missing / np.prod(dataset.mnar_mask_matrix.shape))
    #Count the number of missing entries in the mcar train dataset
    num_missing_mcar_train = np.prod(dataset.mcar_mask_train.shape) - np.sum(dataset.mcar_mask_train)
    print("Number of missing entries in the mcar train dataset:", num_missing_mcar_train / np.prod(dataset.mcar_mask_train.shape)   )
    #Count the number of missing entries in the mcar val dataset
    num_missing_mcar_val = np.prod(dataset.mcar_mask_val.shape) - np.sum(dataset.mcar_mask_val)
    print("Number of missing entries in the mcar val dataset:", num_missing_mcar_val / np.prod(dataset.mcar_mask_val.shape))
    #Count the number of missing entries in the mcar test dataset
    num_missing_mcar_test = np.prod(dataset.mcar_mask_test.shape) - np.sum(dataset.mcar_mask_test)
    print("Number of missing entries in the mcar test dataset:", num_missing_mcar_test / np.prod(dataset.mcar_mask_test.shape))

    # Do histogram of each dataset ratings

    unique, count = np.unique(dataset.mnar_matrix[dataset.mnar_mask_matrix==1], return_counts=True)
    print("unique ratings in mnar dataset:", unique)
    print("counts ratings in mnar dataset:", count)

    unique, count = np.unique(dataset.mcar_train[dataset.mcar_mask_train==1], return_counts=True)
    print("unique ratings in mcar train dataset:", unique)
    print("counts ratings in mcar train dataset:", count)

    unique, count = np.unique(dataset.mcar_val[dataset.mcar_mask_val==1], return_counts=True)
    print("unique ratings in mcar val dataset:", unique)
    print("counts ratings in mcar val dataset:", count)

    unique, count = np.unique(dataset.mcar_test[dataset.mcar_mask_test==1], return_counts=True)
    print("unique ratings in mcar test dataset:", unique)
    print("counts ratings in mcar test dataset:", count)
    