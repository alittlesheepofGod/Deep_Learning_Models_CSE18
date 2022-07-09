import numpy as np 

a = np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1]])

b = np.argmax(a, axis=1)

print('np.argmax(a, axis=1): {0}'.format(b))