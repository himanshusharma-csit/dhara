import tensorflow as tf



def setup_learning_rate(total_steps=None):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4, # This is the peak learning rate
        decay_steps=total_steps,    # Total steps to reach the bottom of the curve
        alpha=0.03                  # Minimum LR (3% of 3e-4 = 9e-6)
    )
    return lr_schedule
