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
X_train = np.resize(X_train, (X_train.shape[0], 72, 1))
X_test = np.resize(X_test, (X_test.shape[0], 72, 1))

# CNN model
def model(n_filters):
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=3, activation='sigmoid', padding='same', input_shape=(72, 1)))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=n_filters, kernel_size=1, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='sigmoid'))

    opt = SGD(lr = 0.01, momentum = 0.9, decay = 0.01)
    # opt = Adagrad()

    model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model
    # adding a pooling layer 

# model = model()
# model.summary()

# hyper-parameters 
n_params = [8, 16, 32, 64, 128, 256]

# Visualization of Results (CNN)
# Let's make a graphical visualization of results obtained by applying CNN to our data 
for p in n_params:
    scores = list()
    for i in range(5):
        model_ = model(p)
        his = model_.fit(X_train, y_train, epochs=5, batch_size=640, validation_data=(X_test, y_test), verbose = 1)
        score = model_.evaluate(X_test, y_test, verbose = 1)
        scores.append(score)
    print(scores, n_params)

# check history of model 
history = his.history
history.keys()

epochs = range(1, len(history['loss']) + 1)
acc = history['accuracy']
loss = history['loss']
val_acc = history['val_accuracy']
val_loss = history['val_loss']

# visualize training and val accuracy
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Loss (CNN)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epochs, loss, label='loss', color='g')
plt.plot(epochs, val_loss, label='val_loss', color='r')
plt.legend()

# Conclusion after CNN Training 

"""
After training our deep CNN model on training data and validating it on validation data, it can be 
interpreted that:

+ model 10 epochs : accuracy : 0.86

-> chon number of filters 
n_params = [8, 16, 32, 64, 128, 256]

"""

