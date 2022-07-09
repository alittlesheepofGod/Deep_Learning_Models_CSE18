# import libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# path to dataset 
PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-14-2018.csv/02-14-2018.csv"

# open dataset csv file by 'pandas'
dataset = pd.read_csv(PATH_TO_DATASET)