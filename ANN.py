import numpy as np 

# import module handling
import handling 

X_train = handling.X_train
X_test = handling.X_test

y_train = handling.y_train
y_test = handling.y_test

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72, 1))
X_test = np.resize(X_test, (X_test.shape[0], 72, 1))

# ANN model 
from sklearn.neural_network import MLPClassifier
def model():
    model = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='sigmoid', solver='adam', max_iter=500)
    return model

model = model()

his = model.fit(X_train, y_train)




