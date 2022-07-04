import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
import numpy as np 

# load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# show data of X_train[0] and label of y_train[0]
plt.matshow(X_train[0])
print(y_train[0])

X_train = X_train / 255
X_test = X_test / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback])

model.get_weights()

# tensorboard --logdir /mnt/d/project-chau/cse-cic-ids2018/tensorboard/log/