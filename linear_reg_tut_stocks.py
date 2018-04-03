import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import quandl,math
import numpy as np
import pickle
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key =  """write your key here"""

df = quandl.get("WIKI/GOOGL")
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])*100/df['Adj. Low']
df['PCT_Change']=(df['Adj. Close']-df['Adj. Open'])*100/df['Adj. Open']
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X = np.array(df.drop(['Label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
Y = np.array(df['Label'])
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
#clf = LinearRegression()
clf = pickle.load(open('linear_regression.pickle','rb'))    

clf.fit(X_train,Y_train)
confidence = clf.score(X_test,Y_test)
forecast_set = clf.predict(X_lately)
style.use('ggplot')
df['Forecast']=np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()