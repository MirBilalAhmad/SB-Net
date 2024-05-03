

import pandas as pd
import numpy as np
import numpy as np
import rdkit
from rdkit import Chem
from sklearn.model_selection import train_test_split

# Read the Data
df = pd.read_csv("final_data.csv")

# Split the data into features (X) and target variable (y)
X = df['processed_products']
y = df['unique_label']

print(y.value_counts())


# Split the data into train and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting sets
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_test:', y_test.shape)

# Concatenate features and labels for the test set
test_data = pd.concat([X_train, y_train], axis=1)
# Save the train data to a CSV file
test_data.to_csv('path/train_data.csv', index=False)

# Concatenate features and labels for the test set
test_data = pd.concat([X_test, y_test], axis=1)

# Save the test data to a CSV file
test_data.to_csv('Path/test_data.csv', index=False)
