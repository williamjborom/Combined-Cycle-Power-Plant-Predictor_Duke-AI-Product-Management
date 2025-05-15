import wget
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Download Data
if not os.path.exists("CCPP_data.csv"):
    URL = "https://storfage.googleapis.com/aipi_datasets/CCPP_data.csv"
    data = wget.download(URL)
else:
    print("Data already downloaded")

#Read data to numpy array
data = np.loadtxt('CCPP_data.csv', delimiter=',', skiprows=1)

#Train / Test Spit
X = data[:,:-2]
y = data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


#Linear Model 1
lm1 = LinearRegression().fit(X_train, y_train)
train_accuracy = lm1.score(X_train, y_train)
print(f"Train accuracy for Linear Model 1 (Basic) is {train_accuracy}")



