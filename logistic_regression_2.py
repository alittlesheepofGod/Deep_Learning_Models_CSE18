import numpy as np 
from sklearn import linear_model

# Reshaped for Logistic function. 
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

# predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(np.array([4.46]).reshape(-1, 1))
print(predicted)

# probability nek 
def logit2prob(logr, x):
    log_odds = logr.coef_ * x + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)
    return (probability)

print(logit2prob(logr, X))

