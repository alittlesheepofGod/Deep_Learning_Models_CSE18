import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

n = 1000 
ratio = .95 # percent of value 1
n_0 = int((1 - ratio) * n)  # number of 
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)   # y is a numpy array including n_0 number of value 0 and n_1 number of value 1

# below are the probabilities obtained from a hypothetical model that always predicts the majority class 
# probability of predicting class 1 is going to be 100% 
y_proba = np.array([1] * n)
y_pred = y_proba > .5 

print(f'accuracy score: {accuracy_score(y, y_pred)}')
cf_mat = confusion_matrix(y, y_pred)
print('Confusion maxtrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')