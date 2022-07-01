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

# import module handling
import handling 

X_train = handling.X_train
X_test = handling.X_test

y_train = handling.y_train
y_test = handling.y_test

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72, 1))
X_test = np.resize(X_test, (X_test.shape[0], 72, 1))

# CNN model
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())

    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    # adding a pooling layer 

model = model()
model.summary()

his = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Visualization of Results (CNN)
# Let's make a graphical visualization of results obtained by applying CNN to our data 
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

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

+ Model was trained on 50 epochs and then on 30 epochs 
+ CNN performed exceptionally well on training data and the accuracy was 99%
+ Model accuracy was down to 83.55% on validation data after 50 iterations, and gave a good accuracy
of 92% after 30 iterations. Thus, it can be interpreted that optimal number of iterations on which this
model can perform are 30.


"""
