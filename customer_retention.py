import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Load the file from the actual file path
file_path = '/Users/hayat/Documents/GitHub/Starbucks_Customer_Retention_Prediction_Project/Starbucks_satisfactory_survey.csv.xls'
customer_data = pd.read_csv(file_path)

print(customer_data.head())
print(customer_data.shape)

# Check for missing values
print("Missing Values:")
print(customer_data.isnull().sum())

# Explore the distribution of numeric variables
print("Numeric Variables Distribution:")
print(customer_data.describe())

# Checking the categorical columns
print(customer_data.dtypes)

# Selected numeric variables
numeric_vars = ['age', 'income', 'visitNo', 'spendPurchase', 'productRate', 'priceRate']

# # Plot histograms
# plt.figure(figsize=(15, 10))
# for i, var in enumerate(numeric_vars, 1):
#     plt.subplot(2, 3, i)
#     sns.histplot(customer_data[var], bins=20, kde=True)
#     plt.title(f'Distribution of {var}')

# plt.tight_layout()
# plt.show()

# # Selected categorical variables
# categorical_vars = ['gender', 'status', 'method', 'location', 'membershipCard']

# # Calculate correlation matrix
# corr_matrix = customer_data.corr()

# # Plot heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

# # Explore relationships with other variables (e.g., 'loyal' vs. 'promoRate')
# sns.boxplot(x='loyal', y='promoRate', data=customer_data)
# plt.title('Loyal vs. Promo Rate')
# plt.show()

# # Pairwise scatter plots
# sns.pairplot(customer_data[['age', 'income', 'spendPurchase', 'productRate', 'priceRate', 'ambianceRate', 'wifiRate', 'serviceRate', 'chooseRate']])
# plt.show()

# # feature importance
# X = customer_data.drop('loyal', axis = 1)
# y = customer_data['loyal']

# # Create a Randome Forest Classifier model
# model = RandomForestClassifier()
# model.fit(X, y)

# # Plot feature importances
# feature_importance = pd.Series(model.feature_importances_, index = X.columns)
# feature_importance.nlargest(10).plot(kind='barh')
# plt.title('Top 10 important features')
# plt.show()

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)

# # Make the prediction
# y_pred = model.predict(X_test)

# # Model evaluation
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# print("Accuracy:", round(accuracy*100,2))
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)

# # Precision: The model is better at predicting class 0 than class 1. For class 0, precision is 0.88, while for class 1, it's only 0.25.
# # Recall: The model has higher recall for class 0 (0.70) than class 1 (0.50).
# # F1-score: The F1-score is a harmonic mean of precision and recall. It provides a balance between precision and recall. The weighted average F1-score is 0.70.

# # True Positive (TP): 7
# # True Negative (TN): 1
# # False Positive (FP): 3
# # False Negative (FN): 1

# # Now we are doing the fine tuning 
# # 1. Feature Engineering:

# # Assuming X_train and X_test are your feature matrices, and y_train and y_test are the corresponding labels
# # Perform feature selection if needed
# selected_features = feature_importance.nlargest(10).index.tolist()
# X_train_selected = X_train[selected_features]
# X_test_selected = X_test[selected_features]

# # 2. Hyperparameter Tuning

# # Assuming you are using a RandomForestClassifier
# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_selected, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# # 3. Cross-Validation

# model = RandomForestClassifier(**best_params)
# cross_val_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
# print("Cross validation scores: ", cross_val_scores)
# print("Mean Accuracy: ", round(np.mean(cross_val_scores)*100,2)) 

# # The mean accuracy is computed from the accuracy scores obtained on each fold. 
# # The mean accuracy provides a more stable estimate of the model's performance compared to looking at the accuracy on individual folds. 