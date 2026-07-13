
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------DATA COLLECTION---------------------------------

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



# feature = data.iloc[13:14,0:20]
# target = data.iloc[32:33,0:20]
# print(feature.info())
# print(feature.shape)
# print(feature.describe())
# print(target.info())
# print(target.shape)
# print(target.describe())

# -----------------------------GRAPH----------------------------
# import matplotlib.pyplot as mp
# mp.plot(feature,target)
# # mp.scatter(feature,target)
# mp.title("progress")
# mp.xlabel('studyhour')
# mp.ylabel('grade')
# mp.grid(True)
# mp.show()