import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier


heart = pd.read_csv('heart.csv')

# Display the first few rows of the dataset
# pd.set_option('display.max_columns', None)
# print(heart.head())
# print(heart.info())
# print(heart.describe())

# Check for missing values
# print(heart.isna().sum())

# # Split the dataset into features and target variable
X = heart.drop(columns='HeartDisease')
y = heart['HeartDisease']

############## KNN CLASSIFIER #########################

# # Convert categorical variables to dummy variables
# X_n = pd.get_dummies(X, drop_first=True)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_n, y, test_size=0.2,
#                                                      random_state=18, stratify=y)

# # Identify numeric columns (before dummies, but after get_dummies all columns are numeric, so you may want to select only original numeric columns)
# original_numeric_cols = X.select_dtypes(include=np.number).columns
# numeric_cols = [col for col in X_n.columns if any(nc in col for nc in original_numeric_cols)]

# # Scale only the numeric columns, leave dummies as-is
# ct = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_cols)
#     ],
#     remainder='passthrough'
# )

# X_train_s = ct.fit_transform(X_train)
# X_test_s = ct.transform(X_test)


# # ## Search for the best number of neighbors

# # recall_scores=[]
# # for k in range(1, 15):
# #     model = KNeighborsClassifier(n_neighbors=k)
# #     model.fit(X_train_s, y_train)
# #     y_pred = model.predict(X_test_s)
# #     recall = recall_score(y_test, y_pred)
# #     recall_scores.append([k, recall])
# # knn_scores = pd.DataFrame(recall_scores, columns=['k', 'recall'])

# # ## Visualise the recall scores
# # sns.lineplot(data=knn_scores, x='k', y='recall')
# # plt.title('Recall Scores for Different k Values')
# # plt.xlabel('Number of Neighbors (k)')
# # plt.ylabel('Recall Score')
# # plt.show()
# # Define the model
# model = KNeighborsClassifier(n_neighbors=7)

# # Fit the model on the training data
# model.fit(X_train_s, y_train)

# # Test the model on the test data
# y_pred = model.predict(X_test_s)
# print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
# print(f"Test accuracy: {model.score(X_test_s, y_test):.2f}")
# print(classification_report(y_test, y_pred))





########################## LOGISTIC REGRESSION #########################




# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
#                                                      random_state=18, stratify=y)

# # Identify numeric and categorical columns
# numeric_cols = X.select_dtypes(include=np.number).columns
# categorical_cols = X.select_dtypes(exclude=np.number).columns

# # Scale the data
# ct = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_cols),
#         ('cat', OneHotEncoder(drop='first'), categorical_cols)
#     ]
# )
# # Transform the training and testing data
# X_train = ct.fit_transform(X_train)
# X_test = ct.transform(X_test)

# # Check for NaN and infinite values
# # print("NaN in X_train:", np.isnan(X_train).any())
# # print("Inf in X_train:", np.isinf(X_train).any())
# # print("NaN in X_test:", np.isnan(X_test).any())
# # print("Inf in X_test:", np.isinf(X_test).any())
# # print("Max value in X_train:", np.max(X_train))
# # print("Min value in X_train:", np.min(X_train))

# # Define the model
# model = LogisticRegression(max_iter=1000, C=1, penalty='l2', solver='liblinear')

# # Hyperparameter tuning using GridSearchCV
# # param_grid = {
# #     'C': [0.001, 0.01, 0.1, 1, 10, 100],
# #     'penalty': ['l1', 'l2'],
# #     'solver': ['liblinear', 'saga']
# # }
# # kf = KFold(n_splits=5, shuffle=True, random_state=18)

# # model_cv = GridSearchCV(model, param_grid, cv=kf, scoring='f1', n_jobs=-1)
# # model_cv.fit(X_train, y_train)
# # print("Best params:", model_cv.best_params_)

# # best_model = model_cv.best_estimator_

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# recall = recall_score(y_test, y_pred)
# print(f"Recall Score: {recall:.2f}")






################## RANDOM FOREST CLASSIFIER #########################

# Define the model
model = RandomForestClassifier(bootstrap=True, max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=100)

X_d = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_d, y, test_size=0.3,
                                                     random_state=18, stratify=y)

# Grid search for hyperparameter tuning
# param_grid = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [None, 10, 20, 30, 40],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }
# kf = KFold(n_splits=7, shuffle=True, random_state=18)

# model_cv = GridSearchCV(model, param_grid, cv=kf, scoring='f1', n_jobs=-1)
# model_cv.fit(X_train, y_train)
# print("Best parameters found:", model_cv.best_params_)   'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 100

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# y_test must be a numpy array for indexing
y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

# Find indices of false negatives: actual 1, predicted 0
false_negative_indices = np.where((y_test_array == 1) & (y_pred_array == 0))[0]

# Get the probabilities for these cases
false_negative_probs = y_proba[false_negative_indices]

# Optionally, print details for inspection
for idx in false_negative_indices:
    print(f"Index: {idx}, True label: {y_test_array[idx]}, Predicted: {y_pred_array[idx]}, Probability: {y_proba[idx]:.3f}")

threshold = 0.45

y_pred_custom = (y_proba >= threshold).astype(int)
# Evaluate the model

print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

sns.heatmap(confusion_matrix(y_test, y_pred_custom), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(ticks=[0.5, 1.5], labels=['No Heart Disease', 'Heart Disease'])
plt.yticks(ticks=[0.5, 1.5], labels=['No Heart Disease', 'Heart Disease'])
plt.tight_layout()

plt.show()