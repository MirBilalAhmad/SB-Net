
# Import Libraries
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem




def get_functional_groups(mol):
    functional_groups = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ['C', 'N', 'O', 'S', 'P']:
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            if 'H' in neighbors:
                neighbors.remove('H')  # Exclude hydrogen atoms
            if neighbors:
                functional_groups.append((symbol, ''.join(sorted(neighbors))))
    return functional_groups

def generate_augmented_smiles(smiles, num_variations=4):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None or mol.GetNumAtoms() <= 1:
        return [smiles]
    
    augmented_smiles_list = []
    for _ in range(num_variations):
        current_mol = Chem.RWMol(mol)
        
        # Get functional groups in the molecule
        functional_groups = get_functional_groups(current_mol)
        if not functional_groups:
            return [smiles]
        
        # Randomly select pairs of functional groups to interchange
        selected_groups = np.random.choice(len(functional_groups), size=2, replace=False)
        group1_index, group2_index = selected_groups[0], selected_groups[1]
        
        # Interchange the positions of the selected functional groups
        for atom in current_mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == functional_groups[group1_index][0]:
                atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(functional_groups[group2_index][0]))
            elif symbol == functional_groups[group2_index][0]:
                atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(functional_groups[group1_index][0]))
        
        # Convert the modified molecule back to SMILES
        augmented_smiles_list.append(Chem.MolToSmiles(current_mol))
    
    return augmented_smiles_list


train_data = pd.read_csv('train_data.csv')

# Apply the augmentation function to each row in the dataset
train_data['processed_products'] = train_data['processed_products'].apply(generate_augmented_smiles)

# Explode the list of augmented SMILES strings into multiple rows
aug_train_data = train_data.explode('processed_products').reset_index(drop=True)


aug_train_data['labels'] = train_data['unique_label']

# Create a DataFrame from the augmented SMILES and labels
aug_train_data = pd.DataFrame({'augmented_smiles': processed_products, 'label': labels})



aug_train_data.to_csv('train_aug_data.csv', index = False)
