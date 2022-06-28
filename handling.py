# import required libraries 
import pandas as pd 
pd.set_option('display.max_columns', None)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# path to dataset 
PATH_TO_DATASET = "/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/processed_first_final_dataset_benign_malicious_of_10_days_2.csv"

# open dataset csv file by 'pandas'
dataset = pd.read_csv(PATH_TO_DATASET)

# get an overview of the data 
dataset.head()
dataset.sample(10)
dataset.tail()

# shape of dataset
print("shape of dataset is : ", dataset.shape)

# identify variables 
dataset.dtypes 
print(dataset.info(verbose=True, max_cols=True, null_counts=True))

# remove object data type in dataset 
# remove Flow ID, Src IP, Dst IP
dataset = dataset.drop(['Flow ID', 'Src IP', 'Dst IP', 'Src Port'], axis = 1)

# print dataset after drop columns with object data type and Src Port column due to many missing values
print(dataset.info(verbose=True, max_cols=True, null_counts=True))

# get count of missing values in the dataset 
print(dataset.isnull().sum())

# print out columns of dataset/dataframe 
print(dataset.columns)

# describe predictors variables 
print(dataset.describe())

