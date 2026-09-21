import tensorflow as tf


# This class holds the Image Encoder of our model
class ImageEncoder(tf.keras.Model):
    def __init__(self, batch_size=None, input_shape=None, encoding_size=None):
        super(ImageEncoder, self).__init__()
        # The input layer that reads the image tensors
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size)
        # The layer that actually encodes the image representation
        self.encoding_layer = tf.keras.layers.Dense(units=encoding_size, activation='relu')
        # The dropout layer to prevent overfitting
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)

    def __call__(self, image_batch_features, training=False):
        x = self.input_layer(image_batch_features)
        x = self.encoding_layer(x)
        x = self.dropout_layer(x, training=training)
        return x


# This class holds the Caption Encoder of our model
class CaptionEncoder(tf.keras.Model):
    def __init__(self, batch_size=None, vocabulary_length=None,
                 encoding_size=None, pretrained_weights=None):
        super(CaptionEncoder, self).__init__()
        # The input layer that reads the caption token indices
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(), batch_size=batch_size)

        if pretrained_weights is not None:
            # Initialize embedding layer with pretrained weights and allow fine-tuning
            self.encoding_layer = tf.keras.layers.Embedding(
                input_dim=vocabulary_length, output_dim=encoding_size,
                mask_zero=True, weights=[pretrained_weights], trainable=True)
            print(f">>> CaptionEncoder initialized with pretrained embeddings "
                  f"(shape: {pretrained_weights.shape}, trainable=True)")
        else:
            # Random initialization
            self.encoding_layer = tf.keras.layers.Embedding(
                input_dim=vocabulary_length, output_dim=encoding_size, mask_zero=True)

        # The dropout layer to prevent overfitting
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.5)

    def __call__(self, caption_batch_embeddings, training=False):
        x = self.input_layer(caption_batch_embeddings)
        x = self.encoding_layer(x)
        x = self.dropout_layer(x, training=training)
        return x
