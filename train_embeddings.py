# Start by training model on give dataset
########################################################

# Import dataset

from pykeen.datasets import DRKG

dataset = DRKG()
dataset.summarize()


# Train embedding model on dataset
########################################################

from pykeen.pipeline import pipeline

pipeline_result = pipeline(
    dataset="DRKG",
    model="TransE",
    # Training configuration
    training_kwargs=dict(
        num_epochs=50,
        use_tqdm_batch=False,
    ),
    # Runtime configuration
    random_seed=1234,
    device="gpu",
)

# Save training triples to directory
pipeline_result.save_to_directory('./DRKG_transe')
