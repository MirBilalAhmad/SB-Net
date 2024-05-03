# Import libraries

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
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


# Read the data


aug_data1 = pd.read_csv('train_aug_data.csv')


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
unique_chars = set(''.join(aug_data1["augmented_smiles"]))

# Sort the unique characters to ensure consistency
unique_chars = sorted(list(unique_chars))

# Apply one-hot encoding to each SMILES string
OHE_pp = []
for smi in aug_data1["augmented_smiles"]:
    OHE_pp.append(smiles_encoder(smi, 300, unique_chars))
OHE_pp = np.asarray(OHE_pp)

print("Shape of one-hot encoded matrix:", OHE_pp.shape)

# Transpose the one-hot encoded matrix
OHE_pp = np.transpose(OHE_pp, (0, 2, 1))

print("Shape of one-hot encoded matrix:", OHE_pp.shape)



# ECFP fingerprints
# Initialize a list to store ECFP representations
ecfp_list = []

for index, row in aug_data1.iterrows():
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

labels_column = aug_data1['label']

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


# *******Model ***************

 from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional, PReLU, GRU, BatchNormalization, concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# Define input layers
input_onehot = Input(shape=(300, 42), name='input_onehot')
input_ecfp = Input(shape=(2048, 1), name='input_ecfp')

# One-hot branch with Convolutional and LSTM layers
conv1d_out_onehot = Conv1D(256, kernel_size=5, activation='PReLU')(input_onehot)
maxpool_out_onehot = MaxPooling1D(pool_size=4)(conv1d_out_onehot)
lstm_out_onehot = Bidirectional(LSTM(512))(maxpool_out_onehot)
onehot_dense = BatchNormalization()(lstm_out_onehot)
onehot_dense = Dropout(0.5)(onehot_dense)

# ECFP branch with Convolutional layer
conv1d_out_ecfp = Conv1D(256, kernel_size=7, activation='PReLU')(input_ecfp)
maxpool_out_ecfp = MaxPooling1D(pool_size=4)(conv1d_out_ecfp)
ecfp_lstm_out = Bidirectional(LSTM(512))(maxpool_out_ecfp)
ecfp_dense = BatchNormalization()(ecfp_lstm_out)
ecfp_dense = Dropout(0.5)(ecfp_dense)

# Concatenate the outputs of both branches
merged = concatenate([onehot_dense, ecfp_dense])

# Additional dense layers with L2 regularization
dense_layer1 = Dense(512, activation='PReLU', kernel_regularizer=l2(0.01))(merged)
dense_layer1 = BatchNormalization()(dense_layer1)
dense_layer1 = Dropout(0.5)(dense_layer1)

dense_layer2 = Dense(1024, activation='PReLU', kernel_regularizer=l2(0.01))(dense_layer1)
dense_layer2 = BatchNormalization()(dense_layer2)
dense_layer2 = Dropout(0.5)(dense_layer2)

# Output layer with 11,852 classes and softmax activation
output_layer = Dense(11852, activation='softmax', name='output')(dense_layer2)

# Create the model
model = Model(inputs=[input_onehot, input_ecfp], outputs=output_layer)

# Compile the model
opt = Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()



# ***************** split data into train and validation ********


from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=15, restore_best_weights=True)

from sklearn.model_selection import train_test_split

OHE_pp_train, OHE_pp_validation, padded_ecfp_input_train, padded_ecfp_input_validation, onehot_encoded_labels_train, onehot_encoded_labels_validation = train_test_split(onehot_products, ecfp_products, labels, test_size = 0.1, random_state = 42)

print(OHE_pp_train.shape, OHE_pp_validation.shape, padded_ecfp_input_train.shape,  padded_ecfp_input_validation.shape, onehot_encoded_labels_train.shape, onehot_encoded_labels_validation.shape)


# ******** Train the model ********
from keras.utils import Sequence
from keras.callbacks import LearningRateScheduler
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from keras.callbacks import ModelCheckpoint
# Define a custom data generator
class DataGenerator(Sequence):
    def __init__(self, onehot_data, ecfp_data, labels, batch_size):
        self.onehot_data = onehot_data
        self.ecfp_data = ecfp_data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.labels))

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_onehot = self.onehot_data[start:end]
        batch_ecfp = self.fcfp_data[start:end]
        batch_labels = self.labels[start:end]

        return [batch_onehot, batch_fcfp], batch_labels

# Create instances of the data generator for training and validation
train_generator = DataGenerator(OHE_pp_train, padded_ecfp_input_train, onehot_encoded_labels_train, batch_size=128)
validation_generator = DataGenerator(OHE_pp_validation, padded_fecfp_input_validation, onehot_encoded_labels_validation, batch_size=256)


# Define the learning rate schedule function
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0001
    else:
        return 0.00001

# Create the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Define other callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint("Retro2_best_weights.h5", save_best_only=True)

# Fit the model using fit
history = model.fit(
    train_generator,
    epochs=200,
    validation_data=validation_generator,
    callbacks=[early_stop, lr_scheduler, checkpoint]
)


# Save the model architecture to a JSON file
model_json = model.to_json()
with open("CNN_Retro.json", "w") as json_file:
    json_file.write(model_json)

# Save the final trained weights to an HDF5 file
model.save_weights("CNN_Retro_final_weights.h5")

