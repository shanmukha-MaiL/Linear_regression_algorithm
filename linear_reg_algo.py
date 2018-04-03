from statistics import mean
import numpy as np
import math,random
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
"""
x = [1,2,3,4,5]
y = [5,4,6,5,6]
x = np.array(x,dtype=np.float64)
y = np.array(y,dtype=np.float64)
"""
def create_dataset(num_dpts,variance,step=2,correlation=False):
    val = 1
    y = []
    for i in range(num_dpts):
        value = val + random.randrange(-variance,variance)
        y.append(value)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation=='neg':
            val-=step
    x = [i for i in range(num_dpts)]
    return np.array(x,dtype=np.float64),np.array(y,dtype=np.float64)
        
def best_slope_and_intercept(x,y):
    m = (mean(x)*mean(y)-mean(x*y))/(math.pow(mean(x),2)-mean(x*x))
    c = mean(y)-m*mean(x)
    return m,c

def squared_error(y_data,y_line):
    return sum((y_line-y_data)**2)

def r_squared_error(y_data,y_line):
    return 1-squared_error(y_data,y_line)/squared_error(y_data,mean(y_line))

x,y = create_dataset(40,10,2,correlation=False)
m,c = best_slope_and_intercept(x,y)
regression_line = [m*i+c for i in x]
"""predict_in = 7
y_predict = m*predict_in+c
plt.scatter(predict_in,y_predict,label='Prediction')"""
plt.scatter(x,y,label='Data')
plt.plot(x,regression_line,label='Regression Line')
plt.legend(loc=4)
plt.show()
print(r_squared_error(y,regression_line))