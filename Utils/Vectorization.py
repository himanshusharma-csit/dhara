import tensorflow as tf


# This method generates a text vectorization layer for the input vocabulary
def generate_text_vectorization(vocabulary=None, max_caption_length=None):
    # Seed the method to remove any non-deterministic weight production
    tf.keras.utils.set_random_seed(seed=9)
    max_tokens = len(vocabulary) + 2
    # Initialize the text vectorization layer
    text_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                           standardize=None,
                                                           split="whitespace",
                                                           output_mode="int",
                                                           pad_to_max_tokens=True,
                                                           vocabulary=vocabulary,
                                                           output_sequence_length=max_caption_length,
                                                           sparse=False)
    return text_vectorization
