from sklearn.model_selection import train_test_split
import pandas as pd
from model import Model

# --------------------------------DATA COLLECTION AND CLEANING-------------------------------------
data = pd.read_csv('data/student-mat.csv')
print(data.shape)
print(data.columns)
print(data.info())


data_encoded = pd.get_dummies(data,drop_first=True)
print(data_encoded.shape)
print(data_encoded.columns)
print(data_encoded.info())

X = data_encoded.drop(columns=['G3'])
y = data_encoded['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#---------------------------MODEL CREATION USING MODEL CLASS----------------------------------------
Linear = Model("sklearn.linear_model","LinearRegression")
DecisionTree = Model("sklearn.tree","DecisionTreeRegressor")
RandomForest = Model("sklearn.ensemble","RandomForestRegressor")
