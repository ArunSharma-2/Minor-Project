#--------------------------------IMPORTING LIBRARIES---------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return rmse, r2

def read_data(location):
    df= pd.read_csv(location)
    # print(df.shape)
    # print(df.columns)
    # print(df.info())

    df_encoded = pd.get_dummies(df,drop_first=True)
    # print(df_encoded.shape)
    # print(df_encoded.columns)
    # print(df_encoded.info())

    return df_encoded


def plot_grade_distribution(df, title):
    plt.figure(figsize=(8, 5))
    df['G3'].hist(bins=15, color='#4f81bd', edgecolor='black')
    plt.title(f'{title} - Final Grade Distribution')
    plt.xlabel('G3')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#c0504d')
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    plt.title(f'{title} - Actual vs Predicted G3')
    plt.xlabel('Actual G3')
    plt.ylabel('Predicted G3')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f'{title} - Feature Importance')
    plt.bar(range(len(importance)), importance[indices], color='#4f81bd', align='center')
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_names, rmses, r2s, title):
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.bar(x - width / 2, rmses, width, label='RMSE', color='#4f81bd')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, r2s, width, label='R2 Score', color='#c0504d')
    ax2.set_ylabel('R2 Score')

    ax1.set_title(f'{title} - Model Comparison')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

##################################FOR STUDENT MAT DATASET##################################

# ---------------------------------DATA COLLECTION---------------------------------

print("----------------------------Student Mat Dataset Results:------------------------------------------")

df = read_data('data/student-mat.csv')
plot_grade_distribution(df, 'Student Mat')

X = df.drop(columns=['G3'])
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#---------------------------------LINEAR REGRESSION---------------------------------

Linear = LinearRegression()
Linear.fit(X_train,y_train)
Linear_pred = Linear.predict(X_test)
print("Linear Regression Results:")
Linear_rmse, Linear_r2 = evaluate_model(Linear_pred, y_test)
plot_predictions(y_test, Linear_pred, 'Linear Regression')

#---------------------------------DECISION TREE REGRESSOR---------------------------------

DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(X_train,y_train)
DecisionTree_pred = DecisionTree.predict(X_test)
print("Decision Tree Regression Results:")
DecisionTree_rmse, DecisionTree_r2 = evaluate_model(DecisionTree_pred, y_test)
plot_predictions(y_test, DecisionTree_pred, 'Decision Tree Regression')

#---------------------------------RANDOM FOREST REGRESSOR---------------------------------

RandomForest = RandomForestRegressor()
RandomForest.fit(X_train,y_train)
RandomForest_pred = RandomForest.predict(X_test)
print("Random Forest Regression Results:")
RandomForest_rmse, RandomForest_r2 = evaluate_model(RandomForest_pred, y_test)
plot_predictions(y_test,RandomForest_pred,"Random Forest Regression")

#-------------------------------------COMPARING MODELS---------------------------------

model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']
rmses = [Linear_rmse, DecisionTree_rmse, RandomForest_rmse]
r2_scores = [Linear_r2, DecisionTree_r2, RandomForest_r2]

print('Model comparison:')
for name, rmse, r2 in zip(model_names, rmses, r2_scores):
    print(f'{name}: RMSE = {rmse:.4f}, R2 = {r2:.4f}')

plot_model_comparison(model_names, rmses, r2_scores, 'Student Mat')

#----------------------------------FEATURE IMPORTANCE FOR BEST MODEL------------------------------

best_index = int(np.argmax(r2_scores))
best_model_name = model_names[best_index]
best_model = [Linear, DecisionTree, RandomForest][best_index]

print(f'Best performing model: {best_model_name} (R2 = {r2_scores[best_index]:.4f})')
plot_feature_importance(best_model, X.columns, f'{best_model_name} on Student Mat')