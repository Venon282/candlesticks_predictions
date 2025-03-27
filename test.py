import tensorflow as tf
import numpy as np

batch, n_sequence, n_features = 5, 10, 3
datas = tf.random.normal((batch, n_sequence, n_features))

print("Original data shape:", datas.shape)

# Create mask (batch, n_sequence)
mask = np.random.randint(0, 2, (batch, n_sequence))
print("Mask shape:", mask.shape)
print("Mask:", mask)

# Convert mask to tf.bool
mask = tf.cast(mask, tf.bool)
print("Mask as boolean:", mask)


# Apply the mask to datas
# We will use tf.boolean_mask on the sequence dimension (axis=1)
datas_masked = tf.ragged.boolean_mask(datas, mask)

# Print out the new shape
print("Masked data shape:", datas_masked.shape)
print(datas)
print(datas_masked)
