import numpy as np
import tensorflow as tf


# Sinusoidal positional encoding for Transformer sequence positions
def get_positional_encoding(max_len, d_model):
    positions = np.arange(max_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


# Transformer-based decoder with self-attention and cross-attention over image features
class TransformerDecoder(tf.keras.Model):
    def __init__(self, batch_size=None, input_shape=None, output_shape=None,
                 num_heads=8, num_layers=2, ff_dim=None, dropout_rate=0.1, max_seq_len=80):
        super(TransformerDecoder, self).__init__()
        self.hidden_units = input_shape
        self.d_model = input_shape
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Positional encoding (precomputed, not trainable)
        self.positional_encoding = get_positional_encoding(max_seq_len, input_shape)

        # Feedforward hidden dimension defaults to 4x the model dimension
        if ff_dim is None:
            ff_dim = input_shape * 4

        # Transformer decoder layers: each has self-attention, cross-attention, and feedforward
        self.self_attention_layers = []
        self.cross_attention_layers = []
        self.ffn_layers = []
        self.layernorm1 = []
        self.layernorm2 = []
        self.layernorm3 = []
        self.dropout_layers = []

        for _ in range(num_layers):
            self.self_attention_layers.append(
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape // num_heads))
            self.cross_attention_layers.append(
                tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape // num_heads))
            self.ffn_layers.append(tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='relu'),
                tf.keras.layers.Dense(input_shape)
            ]))
            self.layernorm1.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.layernorm2.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.layernorm3.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.dropout_layers.append(tf.keras.layers.Dropout(rate=dropout_rate))

        # Output projection to vocabulary
        self.output_layer = tf.keras.layers.Dense(units=output_shape)

    def call(self, decoder_input=None, image_features=None, hidden=None, training=False):
        # Two modes based on decoder_input rank:
        #   rank-2 (batch, d_model)            → autoregressive inference, step by step
        #   rank-3 (batch, seq_len, d_model)   → vectorized training, full sequence at once

        if decoder_input.shape.ndims == 3:
            # Vectorized training mode: process the full input sequence in one pass.
            # All positions share the same causal mask — no growing hidden buffer needed.
            sequence = decoder_input
        else:
            # Autoregressive inference mode: append the new token to the running sequence.
            current_token = tf.expand_dims(decoder_input, axis=1)
            if hidden is not None and hidden.shape[1] > 0:
                sequence = tf.concat([hidden, current_token], axis=1)
            else:
                sequence = current_token

        seq_len = tf.shape(sequence)[1]
        x = sequence + self.positional_encoding[:, :seq_len, :]

        # image_features is (batch, d_model) or (batch, t, d_model) — handle both
        image_context = image_features if image_features.shape.ndims == 3 else tf.expand_dims(image_features, axis=1)

        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask[tf.newaxis, :, :], dtype=tf.bool)

        for i in range(self.num_layers):
            attn_output = self.self_attention_layers[i](
                query=x, key=x, value=x, attention_mask=causal_mask, training=training)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            x = self.layernorm1[i](x + attn_output)

            cross_output = self.cross_attention_layers[i](
                query=x, key=image_context, value=image_context, training=training)
            x = self.layernorm2[i](x + cross_output)

            ffn_output = self.ffn_layers[i](x, training=training)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.layernorm3[i](x + ffn_output)

        if decoder_input.shape.ndims == 3:
            # Return predictions for all positions: (batch, seq_len, vocab_size)
            return self.output_layer(x)

        # Autoregressive: return only the last position's prediction + updated sequence
        predictions = self.output_layer(x[:, -1, :])
        return predictions, sequence

    # Resets hidden state: returns an empty sequence buffer
    def reset_states(self, batch_size=None):
        return tf.zeros((batch_size, 0, self.hidden_units))


# Factory function: always creates a TransformerDecoder
def generate_decoder(batch_size=None, input_shape=None, output_shape=None, **kwargs):
    decoder = TransformerDecoder(batch_size=batch_size, input_shape=input_shape, output_shape=output_shape)
    print(">>> Decoder initialized: Transformer")
    return decoder
