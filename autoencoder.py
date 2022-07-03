import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # model will be trained on GPU 0

import keras 
from matplotlib import pyplot as plt 
import numpy as np 
import gzip 
from keras.models import Model 
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, Dropout, merge, Reshape, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization 
from keras.models import Model, Sequential 
from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.optimizers import Adadelta, RMSprop, SGD, Adam 
from keras import regularizers 
from keras import backend as K 
from tensorflow.keras.utils import to_categorical 

# function to read data file 
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype = np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)
        return data 

# train and test data 
train_data = extract_data('/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/train-images-idx3-ubyte.gz', 60000) # 60 000 images
test_data = extract_data('/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/t10k-images-idx3-ubyte.gz', 10000) # 10 000 images 

# function to read label file 
def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels 

# train and test labels 
train_labels = extract_labels('/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/train-labels-idx1-ubyte.gz', 60000)
test_labels = extract_labels('/mnt/d/project-chau/cse-cic-ids2018/dataset_repo2/t10k-labels-idx1-ubyte.gz', 10000)

# Data Exploration 

# Shapes of training set 
print("Training set (images) shape: {shape}".format(shape=train_data.shape))

# Shapes of test set 
print("Test set (images shape: {shape}".format(shape=test_data.shape))

# Create dictionary of target classes 
label_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H',
    8: 'I', 
    9: 'J',
}

# let's take a look at a couple of the images in your dataset
plt.figure(figsize=[5, 5])

# display the first image in training data 
plt.subplot(121)
curr_img = np.reshape(train_data[10], (28, 28))
curr_lbl = train_labels[10]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# display the first image in testing data 
plt.subplot(122)
curr_img = np.reshape(test_data[10], (28, 28))
curr_lbl = test_labels[10]
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# data preprocessing 
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
train_data.shape, test_data.shape 
train_data.dtype, test_data.dtype 
np.max(train_data), np.max(test_data)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)
np.max(train_data), np.max(test_data)

# partition the data : 80% training and 20% for validation 

from sklearn.model_selection import train_test_split
train_X, valid_X, train_ground, valid_ground = train_test_split(train_data, train_data, test_size=0.2)

batch_size = 64
epochs = 10 
inChannel = 1
x, y = 28, 28 
input_img = Input(shape = (x, y, inChannel))
num_classes = 10

def encoder(input_img):
    # encoder 
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_img) # 28x28x32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    # decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()

# train model 
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

# plot loss function : training loss and validation loss 
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(10)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# save model 
autoencoder.save_weights('autoencoder.h5')

# Segmenting the Fashion Mnist Images 

# Change the labels from categorical to one-hot encoding 
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding 
print('Original label:', train_labels[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# split train data to validation 
train_X, valid_X, train_label, valid_label = train_test_split(train_data, train_Y_one_hot, test_size = 0.2)

# shape of train_X, valid_X, train_label, valid_label 
train_X.shape, valid_X.shape, train_label.shape, valid_label.shape 

# define classification model : exact same encoder part as in autoencoder architecture 
def encoder(input_img):
    # encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)  
    return conv4

# define fully connected layers that will be stacking up with the encoder function 
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out 

encode = encoder(input_img)
full_model = Model(input_img, fc(encode))

for l1, l2 in zip(full_model.layers[0:19], autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())

# check weights of encoder part of new full_model versus weights of encoder part of trained autoencoder 
import unittest 

#
unittest.TestCase.assertEqual(autoencoder.get_weights()[0][1], full_model.get_weights()[0][1], "unequal")

# freeze the encoder layers 
for layer in full_model.layers[0:19]:
    layer.trainable = False 

# let's compile the model!
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# let's print the summary of the model 
full_model.summary()

# train the model 
classify_train = full_model.fit(train_X, train_label, batch_size=64, epochs=10, verbose=1, validation_data=(valid_X, valid_label))

# save classification model
full_model.save_weights('autoencoder_classification.h5')

# re-train model by making the first nineteen layers trainable as True instead of keeping them False!
for layer in full_model.layers[0:19]:
    layer.trainable = True 

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# let's train the entire model for one last time!
classify_train = full_model.fit(train_X, train_label, batch_size=64, epochs=10, verbose=1, validation_data=(valid_X, valid_label))

# save model the last time
full_model.save_weights('classification_complete.h5')

# plot accuracy versus loss
accuracy = classify_train.history['acc']
val_accuracy = classify_train.history['val_acc']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# evaluate model on test set 
test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)

print('Test loss: ', test_eval[0])
print('Test accuracy: ', test_eval[1])

# predict label 
predicted_classes = full_model.predict(test_data)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_labels.shape

correct = np.where(predicted_classes==test_labels)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_labels)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

