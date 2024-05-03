

# Import libraries

import numpy as np
import pandas as pd



df = pd.read_csv("data_processed.csv")

df.shape

df.drop(['id', 'class','reactant','reaction','reaction_template' ], axis = 1, inplace = True)

# Remove Product Redundancy

import pandas as pd
from rdkit.Chem import DataStructs

# Initialize lists to store unique sequences and labels
unique_sequences = []
unique_labels = []


# Iterate through rows of the processed dataset
for index, row in df.iterrows():
    sequence = row['product']
    labels = row['label']
    #morgan_fp = ','.join(str(int(bit)) for bit in row['Morgan_Fingerprint'].ToBitString())

    # Check if the sequence is unique
    if sequence not in unique_sequences:
        unique_sequences.append(sequence)
        unique_labels.append(labels)
        #unique_morgan_fp.append(morgan_fp)

# Create a new DataFrame with unique sequences and labels
unique_df2 = pd.DataFrame({'unique_product': unique_sequences, 'unique_label': unique_labels})

# Save the DataFrame with unique sequences and labels to a new CSV file
unique_df2.to_csv('unique_data.csv', index=False)

# ******* load unique data ********

df1 = pd.read_csv('unique_data.csv')



#***** Index of the longest SMILES string *****


longest_smiles = max(df1["unique_product"], key=len)
longest_smiles_index = df1[df1["unique_product"] == longest_smiles].index.tolist()
print(f"Longest SMILES: {longest_smiles}")
print(f"Contains {len(longest_smiles)} characters, index in dataframe: {longest_smiles_index[0]}.")
smiles_maxlen = len(longest_smiles)


# ************** Index of smallest SMILES string ****

longest_smiles = min(df1["unique_product"], key=len)
longest_smiles_index = df1[df1["unique_product"] == longest_smiles].index.tolist()
print(f"Longest SMILES: {longest_smiles}")
print(f"Contains {len(longest_smiles)} characters, index in dataframe: {longest_smiles_index[0]}.")
smiles_maxlen = len(longest_smiles)


# ***************** Plot to check sequence lengths ******

import matplotlib.pyplot as plt

# Calculate the lengths of each sequence and store them in a new column
df1['Sequence_Length'] = df1['unique_product'].apply(len)

# Check the minimum and maximum sequence lengths
min_length = df1['Sequence_Length'].min()
max_length = df1['Sequence_Length'].max()

print(f"Minimum Sequence Length: {min_length}")
print(f"Maximum Sequence Length: {max_length}")

# Create a horizontal bar plot to visualize the sequence lengths
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
plt.barh(df1.index, df1['Sequence_Length'], color='blue')
plt.xlabel('Sequence Length')
plt.ylabel('Sequence Index')
plt.title('Sequence Lengths (Longest at the Bottom)')
plt.grid(True)
plt.show()


# Trim the unique-products sequences to max-length 250

trimmed_products = []
#trimmed_reactants = []

max_sequence_length = 250  # Maximum allowed sequence length

for index, row in df1.iterrows():
    pp_product = row['unique_product']
    #reactant = row['reactants']
  

    # Trim sequences to maximum length of 300
    if len(pp_product) > max_sequence_length:
        
        pp_product = pp_product[:max_sequence_length]
#     if len(reactant) > max_sequence_length:
#         reactant = reactant[:max_sequence_length]
    
    
    # Skip sequences with length less than 1
    if len(pp_product) < 1:
        #or len(reactant) < 1:
        continue
    
    # Append trimmed data to lists
    trimmed_products.append(pp_product)
    #trimmed_reactants.append(reactant)

# Create a new DataFrame with the trimmed data
trimmed_dp = pd.DataFrame({
    'trim_products': trimmed_products
    #'reactants': trimmed_reactants
})

trimmed_dp['labels'] = df1['unique_label']

trimmed_dp.head()

trimmed_dp.shape

# Save the DataFrame with unique sequences and labels to a new CSV file
unique_df2.to_csv('Trimmed_unique_data.csv', index=False)
