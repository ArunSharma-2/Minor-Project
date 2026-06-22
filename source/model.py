import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm



# ---------------------------------DATA COLLECTION---------------------------------
data = pd.read_csv('dataSet/student-mat.csv')
db = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

x = db.drop('G3', axis=1)
y = db['G3']

# ---------------------------------MODEL TRAINING---------------------------------
model = lm.LinearRegression()
model.fit(x, y)

# ---------------------------------MODEL PREDICTION---------------------------------
predictions = model.predict(x)
print(predictions)

# ---------------------------------MODEL EVALUATION---------------------------------
plt.scatter(db['G1'], db['G3'], color='blue')
plt.scatter(db['G2'], db['G3'], color='red')
plt.xlabel('G1')
plt.ylabel('G3')
plt.title('Linear Regression')
plt.show()