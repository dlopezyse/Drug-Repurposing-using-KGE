# Restore embedding model already trained
import torch

CHOSEN_DISEASE = 'schistosomiasis'
CHOSEN_MODEL = 'ermlp'

# Load embedding model
model_path = './drkg_' + CHOSEN_MODEL + '/trained_model.pkl'
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
