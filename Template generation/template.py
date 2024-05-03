


# Clone rdchiral https://github.com/connorcoley/retrosim/tree/master/rdchiral

# Import libraries

import pandas as pd
from rdchiral import template_extractor
from tqdm import tqdm
import pandas as pd

# Read the data

raw_train = pd.read_csv('raw_train.csv').values
raw_test = pd.read_csv('raw_test.csv').values
raw_val = pd.read_csv('raw_val.csv').values


reactions_train = [{'_id': reaction[0], 'reactants': reaction[2].split('>>')[0], 'products': reaction[2].split('>>')[1]} for reaction in raw_train]
reactions_test = [{'_id': reaction[0], 'reactants': reaction[2].split('>>')[0], 'products': reaction[2].split('>>')[1]} for reaction in raw_test]
reactions_val = [{'_id': reaction[0], 'reactants': reaction[2].split('>>')[0], 'products': reaction[2].split('>>')[1]} for reaction in raw_val]


# Define template extraction function

def extract(reaction):
    try:
        print(reaction)
        return template_extractor.extract_from_reaction(reaction)
    except KeyboardInterrupt:
        print('Interrupted')
        raise KeyboardInterrupt
    except Exception as e:
        print(e)
        return {'reaction_id': reaction['_id']}
    
    
template_train = [extract(reaction) for reaction in tqdm(reactions_train)]
template_test = [extract(reaction) for reaction in tqdm(reactions_test)]
template_val = [extract(reaction) for reaction in tqdm(reactions_val)]

df_template_train = pd.DataFrame(template_train)
df_template_test = pd.DataFrame(template_test)
df_template_val = pd.DataFrame(template_val)

df_template_train.to_csv('template_train.csv', index = False)
df_template_test.to_csv('template_test.csv', index = False)
df_template_val.to_csv('template_val.csv', index = False)



