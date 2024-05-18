import numpy as np
import matplotlib.pyplot as plt


def get_positional_encoding(max_seq_len, d_model):
    # Initialize the positional encoding matrix with zeros
    positional_encoding = np.zeros((max_seq_len, d_model))

    # Calculate the positional encoding values using the given formulas
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))

    return positional_encoding


# Example usage
max_seq_len = 64  # Maximum sequence length
d_model = 256  # Dimension of the model (must be even)
positional_encoding = get_positional_encoding(max_seq_len, d_model)

# Plot the positional encoding for visualization
plt.figure(figsize=(10, 6))
plt.imshow(positional_encoding, cmap='inferno')
plt.xlabel('Embedding Dimension')
plt.ylabel('Sequence Position')
# plt.title('Positional Encoding Matrix')
# plt.colorbar()
plt.savefig('positional_encoding.pdf', format='pdf', bbox_inches='tight')
plt.show()
