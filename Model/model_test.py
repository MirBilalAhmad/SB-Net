
# Import Libraries

import pandas as pd
import numpy as np
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tqdm import tqdm





test_df = pd.read_csv("/path /test_data.csv")

test_df.shape


# Convert augmented_smiles into one hot encoding

import pandas as pd
import numpy as np

# Function defined to create one-hot encoded matrix
def smiles_encoder(smiles, max_len, unique_chars):
    # Create dictionary of the unique char data set
    smi2index = {char: index for index, char in enumerate(unique_chars)}
    # One-hot encoding
    # Zero padding to max_len
    smiles_matrix = np.zeros((len(unique_chars), max_len))
    for index, char in enumerate(smiles[:max_len]):  # Truncate the SMILES string if it's longer than max_len
        smiles_matrix[smi2index[char], index] = 1
    return smiles_matrix


# Extract all unique characters from the augmented SMILES strings
unique_chars = set(''.join(test_df["augmented_smiles"]))

# Sort the unique characters to ensure consistency
unique_chars = sorted(list(unique_chars))

# Apply one-hot encoding to each SMILES string
OHE_pp = []
for smi in test_df["augmented_smiles"]:
    OHE_pp.append(smiles_encoder(smi, 300, unique_chars))
OHE_pp = np.asarray(OHE_pp)

print("Shape of one-hot encoded matrix:", OHE_pp.shape)

# Transpose the one-hot encoded matrix
OHE_pp = np.transpose(OHE_pp, (0, 2, 1))

print("Shape of one-hot encoded matrix:", OHE_pp.shape)



# ECFP fingerprints
# Initialize a list to store ECFP representations
ecfp_list = []

for index, row in test_df.iterrows():
    product_smiles = row['augmented_smiles']
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is not None:
        # Generate ECFP fingerprint (change radius and bitInfo as needed)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=None)
        # Convert ECFP to a list of binary values
        ecfp_bits = list(map(int, ecfp.ToBitString()))
        ecfp_list.append(ecfp_bits)
    else:
        ecfp_list.append([0] * 2048)  # Use zero vector for invalid SMILES

# Create a DataFrame with ECFP representations
ecfp_df3 = pd.DataFrame(ecfp_list, columns=[f'ECFP_{i}' for i in range(2048)])


import numpy as np

# Assuming ecfp_input is a Keras symbolic input
# Convert it to a NumPy array
ecfp_input_array = np.array(ecfp_df3)

# Define the desired kernel size
kernel_size = 2048

# Pad sequences to a minimum length of kernel_size
padded_ecfp_input = np.zeros((len(ecfp_input_array), kernel_size, 1))
for i, sequence in enumerate(ecfp_input_array):
    if len(sequence) >= kernel_size:
        padded_ecfp_input[i, :, 0] = sequence[:kernel_size]
    else:
        padded_ecfp_input[i, :len(sequence), 0] = sequence
        

# Label Processing

labels_column = test_df['label']

# Extract and process the labels
labels = []
for label in labels_column:
    labels.append(label)
    
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# Convert encoded labels to one-hot encoded format
onehot_encoded_cllabels = to_categorical(encoded_labels)

onehot_products = OHE_pp
ecfp_products = padded_ecfp_input
labels = onehot_encoded_cllabels





from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import numpy as np

# 1. Load model architecture from JSON file
with open("CNN-Retro.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# 2. Load trained weights
loaded_model.load_weights("CNN-Retro_final_weights.h5")

# 3. Compile the loaded model (make sure to use the same loss function and optimizer as when you originally trained the model)
opt = Adam()  # Nadam(learning_rate=0.001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 4. Use the loaded model to make predictions on test data
test_predictions = loaded_model.predict([onehot_products, ecfp_products])

# Assuming your test labels are one-hot encoded, you might need to convert predictions to labels
test_labels_pred = np.argmax(test_predictions, axis=1)

# Calculate accuracy (you might need to adjust this depending on how your labels are formatted)
accuracy = np.mean(np.argmax(labels, axis=1) == test_labels_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Calculate top 1% top 3% top 5% and top10% accuracy

top1_accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels, test_predictions, k=1)

# Evaluate the accuracy on your test data
print(f'Top-1 Accuracy: {np.mean(top1_accuracy.numpy())}')

top3_accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels, test_predictions, k=3)

# Evaluate the accuracy on your test data
print(f'Top-3 Accuracy: {np.mean(top3_accuracy.numpy())}')


top5_accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels, test_predictions, k=5)

# Evaluate the accuracy on your test data
print(f'Top-5 Accuracy: {np.mean(top5_accuracy.numpy())}')


top10_accuracy = tf.keras.metrics.top_k_categorical_accuracy(labels, test_predictions, k=10)

# Evaluate the accuracy on your test data
print(f'Top-10 Accuracy: {np.mean(top10_accuracy.numpy())}')