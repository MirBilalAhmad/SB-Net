

# ***** Import libraries *****

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
import pandas as pd
from tqdm import tqdm

# ***** Load trimmed-unique data ****

data1 = pd.read_csv('Trimmed_unique_data.csv')


data1.shape



#****************** Process the SMILES strings in the dataset  **************

def assess_two_letter_elements(data1):


    # Search for unique characters in SMILES strings
    unique_chars = set(data1['unique_product'].apply(list).sum())
    # Get upper and lower case letters only
    upper_chars = []
    lower_chars = []
    for entry in unique_chars:
        if entry.isalpha():
            if entry.isupper():
                upper_chars.append(entry)
            elif entry.islower():
                lower_chars.append(entry)
    #print(f"Upper letter characters {sorted(upper_chars)}")
    #print(f"Lower letter characters {sorted(lower_chars)}")
    
    periodic_elements = ["Ac","Al","Am","Sb","Ar","As","At","Ba","Bk","Be","Bi","Bh","B","Br","Cd","Ca","Cf","C","Ce","Cs","Cl",
                         "Cr","Co","Cn","Cu","Cm","Ds","Db","Dy", "Es", "Er", "Eu", "Fm", "Fl", "F", "Fr", "Gd", "Ga", "Ge", 
                         "Au","Hs","He","Ho","H","In","I","Ir","Fe","Kr","La","Lr","Pb","Li","Lv","Lu","Mg","Mn","Mt","Md","Hg",
                         "Mo","Mc","Nd","Ne","Np","Ni","Nh","Nb","N","No","Og","Os", "O", "Pd", "P", "Pt", "Pu", "Po", "K", "Pr",
                         "Pm","Pa","Ra","Rn","Re","Rh","Rg","Rb","Ru","Rf","Sm","Sc","Sg","Se","Si","Ag","Na","Sr","S","Ta","Tc",
                         "Te","Ts","Tb","Tl","Th","Tm","Sn","Ti","W","U","V","Xe","Yb","Y","Zn","Zr"]

    # The two_char_elements list contains all two letter elements
    # which can be generated by all possible combination of upper x lower characters
    # and are valid periodic elements.
    two_char_elements = []
    for upper in upper_chars:
        for lower in lower_chars:
            ch = upper + lower
            if ch in periodic_elements:
                two_char_elements.append(ch)

    # This list is then reduced to the subset of two-letter elements
    # that actually appear in the SMILES strings, specific to our data set.
    two_char_elements_smiles = set()
    for char in two_char_elements:
        if data1['unique_product'].str.contains(char).any():
            two_char_elements_smiles.add(char)

    return two_char_elements_smiles, unique_chars



elements_found = assess_two_letter_elements(data1)
print(f"\nTwo letter elements found in the data set: {sorted(elements_found)}")


replace_dict = {"Cl": "L", "Br": "R", "Si": "X", "Sn": "Z", 'se': 'E'}




def preprocessing_data(data1, replacement):

    # Print warning if the data set has a 'Sc' element
    if data1['unique_product'].str.contains("Sc").any():
        print(
            'Warning: "Sc" element is found in the data set, since the element is rarely found '
            "in the drugs so we are not converting  "
            'it to single letter element, instead considering "S" '
            'and "c" as separate elements. '
        )

    # Create a new column having processed  SMILES
    data1["processed_products"] = data1["unique_product"].copy()

    # Replace the two letter elements found with one character
    for pattern, repl in replacement.items():
        data1["processed_products"] = data1["processed_products"].str.replace(pattern, repl)

    unique_char = set(data1['processed_products'].apply(list).sum())
    return data1, unique_char
    
    
    
data1, unique_char = preprocessing_data(data1, replace_dict)

data3 = data1[['unique_label', 'processed_products']]


data3.shape

data3.to_csv('final_data.csv', index=False)




