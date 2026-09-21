# Set all the configurations related to the encoder decoder pipeline
import tensorflow as tf
from Config import confs
from Utils.Identifier import Identifier as id
from Utils.LearningRate import setup_learning_rate
from Utils.Dataset import load_preprocessed_dataset
from Utils.ProjectPaths import generate_path_instance
from Utils.Pipeline import initialize_captioning_model
from Utils.FeatureExtractors import get_image_features_shape

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
                               feature_directory=confs[id.feature_directory],
                               word_embedding_choice=confs[id.word_embeddings])

# ----------------------------------------------------------------------------------------------
# The dataset is already preprocessed and saved for now, so we can directly load it
# CODE FOR LOADING THE PRE-SAVED TRAINING DATASET
# ----------------------------------------------------------------------------------------------
dataset = load_preprocessed_dataset(meta_paths=paths, dataset_choice=confs[id.dataset], finetuning=False)
dataset.batch_dataset(batch_size=confs[id.batch_size])
# ----------------------------------------------------------------------------------------------

# Identify the image_feature_shape for CLIP ViT-B/32 (pooled, no spatial)
image_feature_shape = get_image_features_shape(input_shape=confs[id.feature_shape],
                                               extractor_choice=confs[id.feature_extractor],
                                               spatial=confs.get(id.use_patch_tokens, False))

# Now generate the image captioning model for training
model = initialize_captioning_model(paths=paths,
                                    batch_size=confs[id.batch_size],
                                    image_feature_shape=image_feature_shape,
                                    embedding_size=confs[id.embedding_size],
                                    dataset=dataset,
                                    word_embedding_type=confs[id.word_embeddings],
                                    decoder_type=confs[id.decoder_type],
                                    fusion_type=confs.get(id.multimodal_fusion))

batches_per_epoch = len(dataset.x) / confs[id.batch_size]
total_steps = batches_per_epoch * 40

# --- THE SCHEDULE ---
lr_schedule = setup_learning_rate(total_steps=total_steps)

# --- COMPILE ---
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=model.loss_object)

with tf.device(device_name=device):
    model.train_model(paths=paths, start_epoch=0, epochs=40, save=True)
