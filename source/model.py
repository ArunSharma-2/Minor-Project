import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split



# ---------------------------------DATA COLLECTION---------------------------------
data = pd.read_csv('dataSet/student-mat.csv')
db = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

x_train , x_test, y_train, y_test = train_test_split(db[['G1', 'G2', 'studytime', 'failures', 'absences']], db['G3'], test_size=0.3, random_state=42)

# ---------------------------------MODEL TRAINING---------------------------------
model = lm.LinearRegression()
model.fit(x_train, y_train)

# ---------------------------------MODEL PREDICTION---------------------------------
predictions = model.predict(x_test)
print(predictions)

# ---------------------------------MODEL EVALUATION---------------------------------

accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")