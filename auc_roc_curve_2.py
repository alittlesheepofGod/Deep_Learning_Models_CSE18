import numpy as np 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

n = 1000 
ratio = .95 # percent of value 1
n_0 = int((1 - ratio) * n)  # number of 
n_1 = int(ratio * n)

y = np.array([0] * n_0 + [1] * n_1)

y_proba_2 = np.array(
    np.random.uniform(0, .7, n_0).tolist() +
    np.random.uniform(.3, 1, n_1).tolist()
)

y_pred_2 = y_proba_2 > .5 # threashold

print(f'accuracy score: {accuracy_score(y, y_pred_2)}')
cf_mat = confusion_matrix(y, y_pred_2)
print('Confusion matrix')
print(cf_mat)
print(f'class 0 accuracy: {cf_mat[0][0]/n_0}') 
print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')