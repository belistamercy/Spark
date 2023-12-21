import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

file_path = 'C:\\Users\\Belinda\\Downloads\\student_scores.csv'
s_data = pd.read_csv(file_path)

# url = "http://bit.ly/w-data"
# s_data = pd.read_csv(url)
# # url isnt responding i added file as csv

s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

# Plotting
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line)
plt.show()

# predictions
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df)

hours = np.array([[9.25]])
own_pred = regressor.predict(hours.reshape(-1, 1))
print("Predicted Score for 9.25 hours/day:", own_pred[0])
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
