import pandas as pd
import csv
import os
import shutil

# Duplicate the main dataset
shutil.copy('admissionPredict.csv','admissionPredictTemp.csv')

a = int(input("Your GRE Score: "))
b = int(input("Your TOEFL Score: "))
c = float(input("Your University Rating: "))
d = float(input("Number of SOP: "))
e = float(input("Number of LOR: "))
f = float(input("Your CGPA: "))
g = int(input("Any Research Done? ( Yes=1 | No=0): "))

# Writes the user's provided value into csv file
with open('admissionPredictTemp.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([])
    writer.writerow([a, b, c, d, e, f, g, 0.99])

# Imports the dataset
dataset = pd.read_csv('admissionPredictTemp.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,7].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.00125, random_state = 756)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fits Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicts the Test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# Removes the duplicate dataset
os.remove("admissionPredictTemp.csv")