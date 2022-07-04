# import libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# path to dataset 
PATH_TO_DATASET = "/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/02-14-2018.csv"

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
# dataset = dataset.drop(['Flow ID', 'Src IP', 'Dst IP', 'Src Port'], axis = 1)

# print dataset after drop columns with object data type and Src Port column due to many missing values
print(dataset.info(verbose=True, max_cols=True, null_counts=True))

# # get count of missing values in the dataset 
# print(dataset.isnull().sum())

# # print out columns of dataset/dataframe 
# print(dataset.columns)

# # describe predictors variables 
# print(dataset.describe())

# # draw scatter plot 
# dataset.plot.scatter(x = 'Flow Duration', y = 'Tot Fwd Pkts', c = 'Label', colormap='viridis')

# # visualize patterns in the data 
# dataset.corr()

# check the shape of the dataset dataframe
print("shape of dataset : ", dataset.shape)
print("\n")

# check the number of values for labels 
print("the number of values for labels : ", dataset['Label'].value_counts())

# Data Visualizations 

# After getting some useful information about our data, we now make visuals of our data to see how 
# the trend in our data goes like. The visuals include bar plots, distribution plots, scatter plots, etc. 
sns.set(rc={'figure.figsize':(12, 6)})
plt.xlabel('Attack Type')
sns.set_theme()
ax = sns.countplot(x='Label', data = dataset)
ax.set(xlabel='Attack Type', ylabel='Number of Attack and Benign')
plt.show()

# shaping the data for CNN

"""
For applying a convolutional neural network on our data, we will have to follow following steps:

- separate the data of each of the labels
- create a numerical matrix representation of labels
- apply resampling on data so that can make the distribution equal for all labels 
- create X (predictor) and Y (target) variables 
- split the data into train and test sets 
- make data multi-dimensional for CNN
- apply CNN on data --> on separated branch

"""

# encode the column labels
label_encoder = LabelEncoder()
dataset['Label']= label_encoder.fit_transform(dataset['Label'])
dataset['Label'].unique()

# make 3 separate datasets for 3 feature labels 
data_0 = dataset[dataset['Label'] == 0]
data_1 = dataset[dataset['Label'] == 1]
data_2 = dataset[dataset['Label'] == 2]

# make benign feature 
y_0 = np.zeros(data_0.shape[0])
y_benign = pd.DataFrame(y_0)

# make attack feature 
y_1 = np.ones(data_1.shape[0])
y_attack_1 = pd.DataFrame(y_1)
y_2 = np.ones(data_2.shape[0])
y_attack_2 = pd.DataFrame(y_2)

# merging the original dataframe 
X = pd.concat([data_0, data_1, data_2], sort = True)
y = pd.concat([y_benign, y_attack_1, y_attack_2], sort = True)

# Data augmentation 

from sklearn.utils import resample 

data_0_resample = resample(data_0, n_samples = 20000, random_state = 123, replace = True)
data_1_resample = resample(data_1, n_samples = 20000, random_state = 123, replace = True)
data_2_resample = resample(data_2, n_samples = 20000, random_state = 123, replace = True)

train_dataset = pd.concat([data_0_resample, data_1_resample, data_2_resample])
train_dataset.head(2)

# viewing the distribution of intrusion attacks in our dataset 
plt.figure(figsize = (10, 8))
circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Label'].value_counts(), labels = ['Benign', 'FTP-BruteForce', 'SSH-Bruteforce'], colors = ['blue', 'green', 'yellow'])
p = plt.gcf()
p.gca().add_artist(circle)

# making X & Y Variables (CNN)
test_dataset = train_dataset.sample(frac=0.2)
target_train = train_dataset['Label']
target_test = test_dataset['Label']
target_train.unique(), target_test.unique()

y_train = to_categorical(target_train, num_classes = 3)
y_test = to_categorical(target_test, num_classes = 3)

# Data Splicing 
# data into train & test sets. training data used for training model, test data used 
# to check the performance of model on unseen dataset. 
# 80% for training and 20% for testing purpose.
train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis = 1)
test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis = 1)

# making train & test splits 
X_train = train_dataset.iloc[:, : -1].values 
X_test = test_dataset.iloc[:, : -1].values 

# reshape the data for CNN 
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)

