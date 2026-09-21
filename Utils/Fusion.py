import tensorflow as tf
from Utils.Identifier import Identifier as id


# Multimodal Fusion block as described in DHARA (Section 3.3).
#
# Performs cross-modal attention where caption token embeddings (T) serve as
# queries and the visual CLS embedding (v) provides keys and values:
#
#   FusionAttn(T, v) = softmax((T·WQ · (v·WK)^T) / sqrt(dk)) · (v·WV)
#
# Input:
#   visual_features:  (batch, d_model)    — encoded image feature vector
#   caption_sequence: (batch, t, d_model) — caption token embeddings
# Output:
#   Fd:               (batch, t, d_model) — visually-grounded caption embeddings
class CrossAttentionFusion(tf.keras.Model):
    def __init__(self, d_model=512, num_heads=8, dropout_rate=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, visual_features, caption_sequence, training=False):
        # Pooled CLIP features arrive as (batch, d_model); patch tokens arrive as
        # (batch, visual_tokens, d_model). The attention block can consume either.
        visual_context = visual_features
        if visual_features.shape.ndims == 2:
            visual_context = tf.expand_dims(visual_features, axis=1)
        # Text queries attend to visual keys/values
        attn_output = self.cross_attention(
            query=caption_sequence, key=visual_context, value=visual_context,
            training=training)
        attn_output = self.dropout(attn_output, training=training)
        # Residual connection + layer norm to produce Fd
        Fd = self.layer_norm(caption_sequence + attn_output)
        return Fd


# Factory function mirroring the pattern of generate_decoder / generate_attention
def generate_fusion(fusion_type=None, d_model=512, num_heads=8, dropout_rate=0.1):
    if fusion_type is None:
        print(">>> Fusion: None (no multimodal fusion block)")
        return None
    if fusion_type == id.fusion_cross_attention:
        fusion = CrossAttentionFusion(d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        print(f">>> Fusion initialized: {fusion_type}")
        return fusion
    raise ValueError(f"Unknown fusion type: {fusion_type}")
