# import libraries
import keras 

from keras.models import Sequential 
from keras.layers import *
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt 
from keras.datasets import mnist 

# load va chia du lieu 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255
X_test = X_test / 255 

y_train = to_categorical(y_train) # one hot encode
y_test = to_categorical(y_test)   # one hot encode

# xay dung model keras bang Sequential model:
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))  # flat the squared matrix [28, 28] to an array, a squared matrix is an input
model.add(Dense())


model.compile(loss = 'categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

model.summary()

# input x
x = X_train[0]

y = model.fit(x)