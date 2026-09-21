import time
from Config import confs
from Utils.Dataset import initialize_dataset
from Utils.Identifier import Identifier as id
from Utils.ProjectPaths import generate_path_instance


# Start the time counter
start_time = time.time()

# Use the user defined configurations to generate the project paths
paths = generate_path_instance(project_name=confs[id.project_directory],
                               dataset_choice=confs[id.dataset],
                               feature_directory=confs[id.feature_directory],
                               word_embedding_choice=confs[id.word_embeddings])

# Directory from where the file containing all the image names are to be read
dataset_file_path = paths.dataset_file_path()

# ----------------------------------------------------------------------------------------------
# CODE FOR PASCAL HINDI CAPTIONING MODULE
# ----------------------------------------------------------------------------------------------
# Generate the dataset instance using the information from the dataset file path
dataset = initialize_dataset(dataset_name=confs[id.dataset], filepath=dataset_file_path)
# Set the training and testing splits
dataset.split(training_split=0.8, validation_split=0.1, testing_split=0.1)
# Preprocessed and cache the dataset now for future use
dataset.preprocess_data_splits(meta_paths=paths, word_threshold=confs[id.word_threshold])
dataset.save_preprocessed_dataset(meta_paths=paths)
# ----------------------------------------------------------------------------------------------

# Calculate overall time required for execution
total_time = time.time() - start_time

print(f"\n{'='*80}")
print(">>> DATASET PREPROCESSING COMPLETE!")
print(f">>> Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"{'='*80}")