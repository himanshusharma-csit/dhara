import os
import numpy as np
import tensorflow as tf


def clip_preprocess_input(image):
    """Preprocessing for CLIP models (CLIP-specific mean/std normalization)."""
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.48145466, 0.4578275, 0.40821073])
    std = tf.constant([0.26862954, 0.26130258, 0.27577711])
    return (image - mean) / std


def get_preprocess_fn(extractor_choice=None):
    return clip_preprocess_input


# CLIP ViT-B/32 feature extractor
# Pooled mode (spatial=False): feature dim 512 (CLS pooled vector)
# Spatial mode (spatial=True): feature dim (50, 768) (49 patch tokens + 1 CLS token)
def generate_clip_vit_b32_feature_extractor(input_shape=None, spatial=False):
    from transformers import TFCLIPVisionModel

    print(">>> Loading CLIP ViT-B/32 from HuggingFace...")
    clip_vision = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_vision.trainable = False

    class CLIPWrapper(tf.keras.Model):
        def __init__(self, vision_model, img_shape, use_patch_tokens):
            super().__init__(name='CLIP_ViT_B32')
            self.vision_model = vision_model
            self._img_shape = (None,) + tuple(img_shape)
            self.use_patch_tokens = use_patch_tokens

        @property
        def input_shape(self):
            return self._img_shape

        def call(self, inputs, training=False):
            # CLIP expects channels-first (B, C, H, W), TF uses channels-last (B, H, W, C)
            inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
            outputs = self.vision_model(pixel_values=inputs, training=training)
            if self.use_patch_tokens:
                # Return all patch tokens: (batch, num_patches+1, hidden_dim)
                return outputs.last_hidden_state
            return outputs.pooler_output

    model = CLIPWrapper(clip_vision, input_shape, use_patch_tokens=spatial)

    if spatial:
        # 224/32 = 7 -> 7*7 = 49 patches + 1 CLS token = 50 tokens, each 768-dim
        feature_size = (50, 768)
        mode = "spatial (patch tokens)"
    else:
        feature_size = (512,)
        mode = "pooled"
    print(f'>>> CLIP ViT-B/32 feature extractor initialized ({mode})... Feature size: {feature_size}')
    return model, feature_size


def generate_feature_extractor(input_shape=None, extractor_choice=None, spatial=False):
    return generate_clip_vit_b32_feature_extractor(input_shape=input_shape, spatial=spatial)


def get_image_features_shape(input_shape=None, extractor_choice=None, spatial=False):
    if spatial:
        return (50, 768)
    return (512,)


def generate_features_filename(directory_path=None, file_name=None):
    file_name = file_name[:-4] + ".npy"
    return os.path.join(os.path.abspath(directory_path), os.path.relpath(file_name))


# ------------------------------------------------------------------------------------------------------------------
# METHODS RELATED TO FEATURE LOADING ARE HERE
# ------------------------------------------------------------------------------------------------------------------
def load_batch_features(batch_filepaths=None):
    image_features_batch = []
    for image_name in batch_filepaths:
        if isinstance(image_name, bytes):
            filepath = image_name.decode()
        elif isinstance(image_name, str):
            filepath = verify_extension(image_name)
        else:
            raise ValueError(f"Unsupported type for image_name: {type(image_name)}")
        image_features_batch.append(load_features(filepath=filepath))
    return tf.constant(image_features_batch, dtype=tf.float32)


def verify_extension(filepath=None):
    if filepath[-4:] == ".jpg":
        filepath = filepath.replace(".jpg", ".npy")
    return filepath


def load_features(filepath=None):
    image_features = np.load(filepath)
    return image_features
