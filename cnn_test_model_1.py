# cnn model 
from numpy import mean 
from numpy import std 
from numpy import dstack 
from pandas import read_csv 
from matplotlib import pyplot 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.layers.convolutional import Conv1D 
from keras.layers.convolutional import MaxPooling1D 
from keras.utils import to_categorical 

# load a single file as a numpy array 
def load_file(filepath)