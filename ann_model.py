# import libraries
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
import os, re, time, math, tqdm, itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.model_selection import cross_val_score
from sklearnex import patch_sklearn
patch_sklearn()

# For reproducible results
RANDOM_STATE_SEED = 420

# import module handling
import handling 

X_train = handling.X_train
X_test = handling.X_test

y_train = handling.y_train
y_test = handling.y_test

# resize 
X_train = np.resize(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.resize(X_test, (X_test.shape[0], X_train.shape[1]))

# normalizer
scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
scaler = Normalizer().fit(X_test)
X_test = scaler.transform(X_test)

# standard scaler 
s = StandardScaler()
s.fit(X_train)
X_train = s.transform(X_train)
X_test = s.transform(X_test)

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72))
X_test = np.resize(X_test, (X_test.shape[0], 72))

#newcode pca train 
pca=PCA(n_components=72)
pca.fit(X_train)
x_train_pca=pca.transform(X_train)
#newcode pca test
pca.fit(X_test)
x_test_pca=pca.transform(X_test) 

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72))
X_test = np.resize(X_test, (X_test.shape[0], 72))

# ANN model
from sklearn.neural_network import MLPClassifier

def model():  
    model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    return model
    # adding a pooling layer 

model = model()
# model.summary()

his = model.fit(X_train, y_train)

# Visualization of Results (CNN)
# Let's make a graphical visualization of results obtained by applying CNN to our data 
scores = model.score(X_test, y_test)
print(" score ", scores * 100)

# check history of model 
# history = his.history
# history.keys()

# epochs = range(1, len(history['loss']) + 1)
# acc = history['accuracy']
# loss = history['loss']
# val_acc = history['val_accuracy']
# val_loss = history['val_loss']

# # visualize training and val accuracy
# plt.figure(figsize=(10, 5))
# plt.title('Training and Validation Loss (CNN)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(epochs, loss, label='loss', color='g')
# plt.plot(epochs, val_loss, label='val_loss', color='r')
# plt.legend()

# Conclusion after CNN Training 

"""
After training our deep CNN model on training data and validating it on validation data, it can be 
interpreted that:

+ Model was trained on 10 epochs
+ CNN performed exceptionally well on training data and the accuracy was 99%
+ Model accuracy was down to 83.55% on validation data after 50 iterations, and gave a good accuracy
of 92% after 30 iterations. Thus, it can be interpreted that optimal number of iterations on which this
model can perform are 30. --> ... 


"""