# import libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# path to dataset 
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-14-2018.csv/02-14-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-15-2018.csv/02-15-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-16-2018.csv/02-16-2018.csv"

""" because 02-20-2018.csv is too large, it is divided into 8 files  """

# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-20-2018.csv/02-20-2018-01.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-21-2018.csv/02-21-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-22-2018.csv/02-22-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-28-2018.csv/02-28-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/03-01-2018.csv/03-01-2018.csv"
# PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/03-02-2018.csv/03-02-2018.csv"

PATH_TO_DATASET = "/mnt/d/project/dataset/cse-cic-ids2018/02-28-2018.csv/02-28-2018.csv"

# open dataset csv file by 'pandas'
dataset = pd.read_csv(PATH_TO_DATASET)

