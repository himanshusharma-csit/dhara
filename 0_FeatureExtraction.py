import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from Config import confs
from Utils.Dataset import initialize_dataset
from Utils.Identifier import Identifier as id
from Utils.ProjectPaths import generate_path_instance
from Utils.FeatureExtractors import generate_feature_extractor, get_preprocess_fn

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f">>> GPU Detected: {gpus[0].name}")
    except RuntimeError as e:
        print(f">>> GPU Configuration Error: {e}")
else:
    print(">>> WARNING: No GPU detected, will use CPU")

device = '/GPU:0' if gpus else '/CPU:0'
# ============================================================================

# Use the user defined configurations to generate the project paths
paths = generate_path_instance(project_name=confs[id.project_directory],
                               dataset_choice=confs[id.dataset],
                               feature_directory=confs[id.feature_directory])

dataset_file_path = paths.dataset_file_path()
image_directory = paths.images_abs_path()
features_directory = os.path.abspath(paths.features_abs_path())
os.makedirs(features_directory, exist_ok=True)

# Generate the dataset instance
dataset = initialize_dataset(dataset_name=confs[id.dataset], filepath=dataset_file_path)
dataset.split(training_split=0.8, validation_split=0.1, testing_split=0.1)

# Generate the feature extractor
image_encoder, _ = generate_feature_extractor(input_shape=confs[id.feature_shape],
                                              extractor_choice=confs[id.feature_extractor],
                                              spatial=confs.get(id.use_patch_tokens, False))

# Get input specifications once
_, x, y, channels = image_encoder.input_shape
preprocess_fn = get_preprocess_fn(confs[id.feature_extractor])
batch_size = confs[id.batch_size]


def load_and_preprocess(image_path):
    """Load and preprocess a single image using TF ops (runs in parallel)."""
    image_file = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_file, channels=channels)
    image = tf.image.resize(image, (x, y))
    image = preprocess_fn(image)
    return image


def feature_extraction_batched(image_directory, feature_directory, feature_extractor,
                               image_list):
    num_images = len(image_list)
    num_batches = (num_images + batch_size - 1) // batch_size

    # Build full paths for tf.data pipeline
    full_paths = [os.path.join(image_directory, name) for name in image_list]

    # Create tf.data pipeline with parallel loading and prefetching
    path_dataset = tf.data.Dataset.from_tensor_slices(full_paths)
    image_dataset = path_dataset.map(load_and_preprocess,
                                     num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f">>> Processing {num_images} images in {num_batches} batches (batch_size={batch_size})")

    start_time = time.time()
    img_idx = 0
    for batch_tensor in tqdm(image_dataset, total=num_batches,
                             desc="EXTRACTING FEATURES >>> ", ascii=False, ncols=100):
        # Extract features on GPU
        batch_features = feature_extractor(batch_tensor, training=False)

        # Convert entire batch to numpy — preserves spatial dims (H, W, C) if present
        features_np = batch_features.numpy()
        current_batch_size = batch_tensor.shape[0]

        # Save features
        for i in range(current_batch_size):
            file_name = image_list[img_idx][:-4] + ".npy"
            np.save(os.path.join(feature_directory, file_name), features_np[i])
            img_idx += 1

    elapsed_time = time.time() - start_time
    print(f">>> Extracted {num_images} features in {elapsed_time:.2f}s ({num_images / elapsed_time:.2f} images/sec)")


# Combine all image lists to process in one pass
all_image_names = (dataset.training_image_names +
                   dataset.testing_image_names +
                   dataset.validation_image_names)

print(f"\n{'='*80}")
print(f">>> STARTING FEATURE EXTRACTION")
print(f">>> Device: {device}")
print(f">>> Total images: {len(all_image_names)}")
print(f"{'='*80}\n")

overall_start_time = time.time()

feature_extraction_batched(image_directory=image_directory,
                           feature_directory=features_directory,
                           feature_extractor=image_encoder,
                           image_list=all_image_names)

total_time = time.time() - overall_start_time

print(f"\n{'='*80}")
print(">>> FEATURE EXTRACTION COMPLETE!")
print(f">>> Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f">>> Average: {total_time / len(all_image_names):.3f}s per image")
print(f">>> Throughput: {len(all_image_names) / total_time:.2f} images/sec")
print(f"{'='*80}")
