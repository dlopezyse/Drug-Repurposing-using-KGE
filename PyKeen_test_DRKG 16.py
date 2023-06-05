
########################################################
# Import dataset
########################################################

import pandas as pd
from pykeen.datasets import DRKG

dataset = DRKG()

# Create triples of Training set
tf_train = dataset.training

# Create triples of Validation set
tf_val = dataset.validation

# Create triples of Testing set
tf_test = dataset.testing

########################################################
# Query
########################################################

# Convert triples to dataframe for querying

# df_train_query = pd.DataFrame(dataset.training.triples, columns=['head', 'relation', 'tail'])

# print(df_train_query[df_train_query['tail'].str.contains('MESH:D012552')])

# print(df_train_query[df_train_query['head'].str.contains('MESH:C403507')])

# print(df_train_query.loc[(df_train_query['head'] == 'Compound::DB00706') & (df_train_query['tail'].str.contains('Atc::G04CA52'))])

# # Side effects of tamoxifen in DrugBank
# print(df_train_query.loc[(df_train_query['head'] == 'Compound::DB00706') & (df_train_query['tail'].str.contains('Side Effect'))])

# # Number of side effects of tamoxifen in DrugBank
# len(df_train_query.loc[(df['head'] == 'Compound::DB00706') & (df_train_query['tail'].str.contains('Side Effect'))])


############################################################
# Restore Embedding model already trained on Google Colab
############################################################
import torch

CHOSEN_DISEASE = 'schistosomiasis'
CHOSEN_MODEL = 'ermlp'

# Load embedding model
model_path = 'C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/drkg_' + CHOSEN_MODEL + '/trained_model.pkl'
emb_model = torch.load(model_path, map_location=torch.device('cpu')) 

# Perform head prediction and check if it exists in training  and testing sets
testing_mapped_triples = tf_test.mapped_triples.to(emb_model.device)

# Dictionary of disease MESH ID codes
disease_code = {'dengue': 'Disease::MESH:D003715',
                'chagas': 'Disease::MESH:D014355',
                'malaria': 'Disease::MESH:D008288',
                'leishmaniasis': 'Disease::MESH:D007896',
                'yellowfever': 'Disease::MESH:D015004',
                'filariasis':'Disease::MESH:D004605',
                'schistosomiasis':'Disease::MESH:D012552'}

emb_model.get_head_prediction_df("GNBR::T::Compound:Disease", disease_code[CHOSEN_DISEASE], triples_factory=tf_train, testing=testing_mapped_triples)
prediction_df = emb_model.get_head_prediction_df("GNBR::T::Compound:Disease", disease_code[CHOSEN_DISEASE], triples_factory=tf_train, testing=testing_mapped_triples)

########################################################
# Pre-processing
########################################################

# Create column to define if prediction result is a compound or not (using list comprehension)
prediction_df['is_compound'] = ['yes' if 'Compound' in c else 'no' for c in prediction_df['head_label']]

# Create column to identify data source of prediction result (using for loop)
data_source  = []
for row in prediction_df['head_label']:
    if 'MESH' in row:
        data_source.append('mesh')
    elif 'CHEBI' in row:
        data_source.append('chebi')
    elif 'DB' in row:
        data_source.append('db')
    elif 'CHEMBL' in row:
        data_source.append('chembl')
    elif 'brenda' in row:
        data_source.append('brenda')
    else:
        data_source.append('na')
prediction_df['data_source'] = data_source 

# Create a column with the compound ID (using for loop)
comp_id  = []
for x in prediction_df['head_label'].str.split(':', 4):
    comp_id.append(x[-1])
prediction_df['compound_id'] = comp_id

# Save compound IDs on a new 'compound_id' column only if it's a compound on column 'is_compound'
prediction_df['compound_id2'] = prediction_df['compound_id'].where(prediction_df['is_compound'] == 'yes', '-')

# Map IDs with external sources and bring compound names
########################################################


# MESH IDs
#-----------------------------

mesh_data = pd.read_csv("C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/mesh_name_pubmed.txt", sep='\t', names=['compound_id2', 'mesh_name'], header=None)
mesh = mesh_data.copy()

# Remove duplicates
mesh = mesh.drop_duplicates(subset='compound_id2', keep="first")

# Incorporate additional IDs
# taken from: https://github.com/OHDSI/KnowledgeBase/blob/master/LAERTES/terminology-mappings/RxNorm-to-MeSH/mesh-to-rxnorm-standard-vocab-v5-NEW%20FILE%20(tabbed%20delimited).csv

mesh2 = pd.read_csv("C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/mesh_map_list_github_simple.csv")

# Merge tables
prediction_df = pd.merge(prediction_df, mesh, on ='compound_id2', how ='left')
prediction_df['mesh_name'].fillna('-', inplace=True)

prediction_df = pd.merge(prediction_df, mesh2, on ='compound_id2', how ='left')
prediction_df['mesh_name2'].fillna('-', inplace=True)


# DrugBank IDs
#-----------------------------

# Read DrugBank file
db = pd.read_excel('C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/DrugBank_names.xlsx', engine='openpyxl')

# Merge tables
prediction_df = pd.merge(prediction_df, db, on ='compound_id2', how ='left')
prediction_df['db_name'].fillna('-', inplace=True)


# CHEBI IDs
#-----------------------------

chebi = pd.read_csv("C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/chebi_names.csv", sep=";")

# Set compound id to object to perform merge operation
chebi['compound_id2'] = chebi['compound_id2'].astype(str)

# Merge tables
prediction_df = pd.merge(prediction_df, chebi, on ='compound_id2', how ='left')
prediction_df['chebi_name'].fillna('-', inplace=True)


# CHEMBL IDs
#-----------------------------

# chembl_name_pubmed.txt obtained from: https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi
# Le pasé el listado de CHEMBL IDs de DRKG y me devolvió sus nombres

chembl_data = pd.read_csv("C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/chembl_name_pubmed.txt", sep='\t', names=['compound_id2', 'chembl_name'], header=None)
chembl = chembl_data.copy()

# Merge tables
prediction_df = pd.merge(prediction_df, chembl, on ='compound_id2', how ='left')
prediction_df['chembl_name'].fillna('-', inplace=True)


# BRENDA IDs
#-----------------------------

brenda_data = pd.read_csv("C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/DrugBank/brenda_list.csv", sep=";")
brenda = brenda_data.copy()

# Set compound id to object to perform merge operation
brenda['compound_id2'] = brenda['compound_id2'].astype(str)

# Merge tables
prediction_df = pd.merge(prediction_df, brenda, on ='compound_id2', how ='left')
prediction_df['brenda_name'].fillna('-', inplace=True)


# TO DO: PubChem IDs, etc
########################################################


# Create unique compound name from all IDs
import numpy as np

prediction_df['final_comp_name'] = np.where(prediction_df.mesh_name.ne('-'), prediction_df.mesh_name, 
                                    np.where(prediction_df.mesh_name2.ne('-'), prediction_df.mesh_name2, 
                                    np.where(prediction_df.db_name.ne('-'), prediction_df.db_name,         
                                    np.where(prediction_df.chebi_name.ne('-') & prediction_df.data_source.eq('chebi'), prediction_df.chebi_name,
                                    np.where(prediction_df.chembl_name.ne('-'), prediction_df.chembl_name,
                                    np.where(prediction_df.brenda_name.ne('-') & prediction_df.data_source.eq('brenda'), prediction_df.brenda_name,         
                                             '-'))))))

# Define destination folder for prediction file
prediction_path = 'C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/drkg_' + CHOSEN_MODEL

import os

file_name = CHOSEN_MODEL + '_' + CHOSEN_DISEASE

# Create dataframe with final prediction results and their compound names
# prediction_df.to_csv(os.path.join(prediction_path, file_name + '_names.csv'), sep=";")

########################################################
# Test model results against clinical trials
########################################################

import pandas as pd

# Create column with compound name to lower case
prediction_df['final_comp_name_lc'] = prediction_df['final_comp_name'].str.lower()

# Load clinical trials data
clinical_trials = pd.read_excel('C:/Users/CynYDie/Desktop/Austral_2/TESIS/KG/DRKG/pykeen/DRKG/dataset/Clinical_trials/clinical_trials_' + CHOSEN_DISEASE + '.xlsx', engine='openpyxl')

# Match compounds of model against compounds in clinical trials. If match, 1. If not, 0. Create a column with the result
prediction_df['comp_in_trial'] = prediction_df['final_comp_name_lc'].isin(clinical_trials['drug_lc']).astype(int)

# Create dataframe with result
prediction_df.to_csv(os.path.join(prediction_path, file_name + '_prediction.csv'), sep=";")