#--------------------------------IMPORTING LIBRARIES---------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------DATA COLLECTION---------------------------------

data = pd.read_csv('data/student-mat.csv')
print(data.shape)
# print(data.columns)
# print(data.info())


data_encoded = pd.get_dummies(data,drop_first=True)
print(data_encoded.shape)
print(data_encoded.columns)
# print(data_encoded.info())

X = data_encoded.drop(columns=['G3'])
y = data_encoded['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#---------------------------------LINEAR REGRESSION---------------------------------

Linear = LinearRegression()
Linear.fit(X_train,y_train)
Linear_pred = Linear.predict(X_test)
Linear_score = Linear.score(X_test,y_test)
print("Linear Regression Score:", Linear_score)

#---------------------------------DECISION TREE REGRESSOR---------------------------------

DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(X_train,y_train)
DecisionTree_pred = DecisionTree.predict(X_test)
DecisionTree_score = DecisionTree.score(X_test,y_test)
print("Decision Tree Regression Score:", DecisionTree_score)

#---------------------------------RANDOM FOREST REGRESSOR---------------------------------

RandomForest = RandomForestRegressor()
RandomForest.fit(X_train,y_train)
RandomForest_pred = RandomForest.predict(X_test)
RandomForest_score = RandomForest.score(X_test,y_test)
print("Random Forest Regression Score:", RandomForest_score)