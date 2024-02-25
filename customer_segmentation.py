import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

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

# Selected categorical variables
categorical_vars = ['gender', 'status', 'method', 'location', 'membershipCard']

# Calculate correlation matrix
# corr_matrix = customer_data.corr()

# feature importance
X = customer_data.drop('loyal', axis = 1)
y = customer_data['loyal']

# Create a Randome Forest Classifier model
model = RandomForestClassifier()
model.fit(X, y)

feature_importance = pd.Series(model.feature_importances_, index = X.columns)
feature_importance.nlargest(10).plot(kind='barh')
selected_features = feature_importance.nlargest(10).index.tolist()

# Extract features for segmentation
segmentation_features = customer_data[selected_features]

# Choose the number of clusters (you may need to adjust this based on your data)
num_clusters = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(segmentation_features)

# Visualize the clusters
sns.scatterplot(x=selected_features[0], y=selected_features[1], hue='cluster', data=customer_data, palette='viridis')
plt.title('Customer Segmentation using K-Means Clustering')
plt.show()