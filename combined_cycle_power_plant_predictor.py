import wget
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

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
kf = KFold(n_splits=10, shuffle=True, random_state=10)

#Linear Model 1
lm1 = LinearRegression().fit(X_train, y_train)
train_accuracy = lm1.score(X_train, y_train)
print(f"Train accuracy for Linear Model 1 (Basic) is {train_accuracy}")
cv_scores = cross_val_score(lm1, X_train, y_train, cv=kf)
print(f"Average 10-Fold CV Score for Linear Model 1 (Basic) is {cv_scores.mean()}")
test_accuracy_lm1 = lm1.score(X_test, y_test) #Not looking at this until final model choice

#Linear Model 2
poly_2_transform = PolynomialFeatures(degree=2)
X_train_poly2 = poly_2_transform.fit_transform(X_train)
lm2 = LinearRegression().fit(X_train_poly2, y_train)
train_accuracy = lm2.score(X_train_poly2, y_train)
print(f"Train accuracy for Linear Model 2 (Poly) is {train_accuracy}")
cv_scores = cross_val_score(lm1, X_train, y_train, cv=kf)
print(f"Average 10-Fold CV Score for Linear Model 2 (Poly) is {cv_scores.mean()}")
X_test_poly2 = poly_2_transform.fit_transform(X_test) #need to transform test data
test_accuracy_lm2 = lm2.score(X_test_poly2, y_test) #Not looking at this until final model choice

#Random Forest
rf_model = RandomForestRegressor(max_depth=4, random_state=0)
rf = rf_model.fit(X_train, y_train)
train_accuracy = rf.score(X_train, y_train)
print(f"Train accuracy for the Random Forest model is {train_accuracy}")
test_accuracy_rf = rf.score(X_test, y_test) #Not looking at this until final model choice

