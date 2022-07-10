# import libraries
from turtle import clear
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
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(72, 1)))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3, activation='softmax'))

    opt = SGD(lr = 0.01, momentum = 0.9, decay = 0.01)
    # opt = Adagrad()

    model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model
    # adding a pooling layer 

model = model()
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose = 0)

# Visualization of Results (CNN)
# Let's make a graphical visualization of results obtained by applying CNN to our data 
scores = model.evaluate(X_test, y_test, verbose = 1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# epochs = range(1, len(history['loss']) + 1)
# acc = history['accuracy']
# loss = history['loss']
# val_acc = history['val_accuracy']
# val_loss = history['val_loss']

# draw configure matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
y_pred = model.predict(X_test)
# labels = ["Benign", "DoS attacks-GoldenEye", "DoS attacks-Slowloris"]
# labels = ["Benign", "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk"]
labels = ["Benign", "DoS attacks-LOIC-HTTP"]

# convert to categorical 
from keras.utils.np_utils import to_categorical
y_predict = to_categorical(np.argmax(y_pred, 1), dtype="int64")
# convert one-hot encoding to integer
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# visualize training and val accuracy
# plt.figure(figsize=(10, 5))
# plt.title('Training and Validation Loss (CNN)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.plot(epochs, loss, label='loss', color='g')
# plt.plot(epochs, val_loss, label='val_loss', color='r')
# plt.legend()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# print out accuracy for each class
matrix = confusion_matrix(y_test, y_predict)
# print("accuracy of benign, DoS attacks-SlowHTTPTest, DoS attacks-Hulk")
print("accuracy of benign, DoS attacks-LOIC-HTTP")
print(matrix.diagonal()/matrix.sum(axis=1))

# print out False Alarm Rate 
print("False Alarm Rate is : ")
FAR = 0
for i in range(1, len(cm[0])):
    FAR += cm[0][i]
FAR = FAR / (cm[0][0] + FAR)
print(FAR)

# print detection rate
print("Detection Rate is : ")
DTrate = 0
for i in range(1, len(cm)):
    for j in range(0, len(cm[i])):
        DTrate += cm[i][j]

sum = 0
for i in range(1, len(cm)):
    sum += cm[i][i]

DTrate = sum / DTrate 

print(DTrate)

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

