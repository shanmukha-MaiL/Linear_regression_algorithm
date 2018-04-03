import pickle
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
with open('linear_regression.pickle','wb') as f:
    pickle.dump(clf,f)