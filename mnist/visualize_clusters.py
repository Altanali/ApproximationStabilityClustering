import numpy as np
import matplotlib.pyplot as plt

# Load the data
features = np.load('./out/bottleneck_features.npy')
labels = np.load('./out/labels.npy')

# Check if features are 2D
if features.shape[1] != 2:
    raise ValueError("Features are not 2D.")

# Plotting
plt.figure(figsize=(8, 6))
for label in np.unique(labels):
    # Extract features for each label
    indices = labels == label
    label_features = features[indices]

    # Plot each class with a different color and label
    plt.scatter(label_features[:, 0], label_features[:, 1], label=f'Label {label}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('2D Bottleneck Features Visualization')
plt.legend()
plt.grid(True)
plt.show()

