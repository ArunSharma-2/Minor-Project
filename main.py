#--------------------------------IMPORTING LIBRARIES---------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

#-----------------------------------IMPORTING MODELS---------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#-------------------------------EVALUATION FUNCTION---------------------------------

def evaluate_model(model_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, model_pred))
    print("RMSE:", rmse)
    r2 = r2_score(y_test, model_pred)
    print("R2 Score:", r2, "\n")

def read_data(location):
    df= pd.read_csv(location)
    print(df.shape)
    # print(df.columns)
    # print(df.info())

    df_encoded = pd.get_dummies(df,drop_first=True)
    print(df_encoded.shape)
    # print(df_encoded.columns)
    # print(df_encoded.info())

    return df_encoded

##################################FOR STUDENT MAT DATASET##################################
# ---------------------------------DATA COLLECTION---------------------------------

print("----------------------------Student Mat Dataset Results:------------------------------------------")

df_mat = read_data('student-mat.csv')

X = df_mat.drop(columns=['G3'])
y = df_mat['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#---------------------------------LINEAR REGRESSION---------------------------------

Linear = LinearRegression()
Linear.fit(X_train,y_train)
Linear_pred = Linear.predict(X_test)
print("Linear Regression Results:")
evaluate_model(Linear_pred, y_test)

#---------------------------------DECISION TREE REGRESSOR---------------------------------

DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(X_train,y_train)
DecisionTree_pred = DecisionTree.predict(X_test)
print("Decision Tree Regression Results:")
evaluate_model(DecisionTree_pred, y_test)

#---------------------------------RANDOM FOREST REGRESSOR---------------------------------

RandomForest = RandomForestRegressor()
RandomForest.fit(X_train,y_train)
RandomForest_pred = RandomForest.predict(X_test)
print("Random Forest Regression Results:")
evaluate_model(RandomForest_pred, y_test)

#####################################FOR STUDENT POR DATASET##################################
# ---------------------------------DATA COLLECTION---------------------------------

print("----------------------------Student Por Dataset Results:------------------------------------------")

df_por = read_data('student-por.csv')

X = df_por.drop(columns=['G3'])
y = df_por['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#---------------------------------LINEAR REGRESSION---------------------------------

Linear = LinearRegression()
Linear.fit(X_train,y_train)
Linear_pred = Linear.predict(X_test)
print("Linear Regression Results:")
evaluate_model(Linear_pred, y_test)

#---------------------------------DECISION TREE REGRESSOR---------------------------------

DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(X_train,y_train)
DecisionTree_pred = DecisionTree.predict(X_test)
print("Decision Tree Regression Results:")
evaluate_model(DecisionTree_pred, y_test)

#---------------------------------RANDOM FOREST REGRESSOR---------------------------------

RandomForest = RandomForestRegressor()
RandomForest.fit(X_train,y_train)
RandomForest_pred = RandomForest.predict(X_test)
print("Random Forest Regression Results:")
evaluate_model(RandomForest_pred, y_test)