'''
    Main Script for Gender, Age, and Ethnicity identification on the cleaned UTK Dataset.

    The dataset can be found here (https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv)

    I implemented a MultiLabelNN that performs a shared high-level feature extraction before creating a 
    low-level neural network for each classification desired (gender, age, ethnicity).
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
# torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# Custom imports
from CustomUTK import UTKDataset
from MultNN import TridentNN

'''
    Add arguments for argparse

    Arguments:
     - epochs : (int) Number of epochs to train
     - lr : (float) Learning rate for the model
     - pre-trained : (bool) whether or not to load the pre-trained model
'''
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--num_epochs', default=20, type=int, help='(int) number of epochs to train the model')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='(float) learning rate of optimizer')
parser.add_argument('-pt', '--pre-trained', default=True, type=bool, help='(bool) whether or not to load the pre-trained model')



def init_trident():
    # Configure the device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse the arguments
    args = parser.parse_args()

    # Read in the data and store in train and test dataloaders
    train_loader, test_loader, class_nums = read_data()

    # Load the model and optimizer
    tridentNN = TridentNN(class_nums['age_num'], class_nums['gen_num'], class_nums['eth_num'])
    

    # Load and test the pre-trained model
    checkpoint = torch.load('/checkpoints/tridentNN_epoch20.pth.tar')
    tridentNN.load_state_dict(checkpoint['state_dict'])
    test(test_loader, tridentNN)

if __name__ == '__main__':
    main()
