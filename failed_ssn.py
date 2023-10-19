import numpy as np
#!pip install tensorflow


class ConvolutionalESLSNNs:
    def __init__(self, input_shape, num_classes, simulation_time, leaky_factor, Titer, alpha):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.simulation_time = simulation_time
        self.leaky_factor = leaky_factor
        self.threshold = 1
        self.weight_matrix = None
        self.mask_matrix = None
        self.Titer=Titer
        self.alpha=alpha

    def initialize_weights(self):
        # Initialize the weight matrix with random values
        self.weight_matrix = np.random.randn(self.num_classes, np.prod(self.input_shape))

    def initialize_mask(self):
        # Initialize the mask matrix with ones
        self.mask_matrix = np.ones_like(self.weight_matrix)

    def euler_integration(self, inputs):
        # Perform Euler integration to update membrane potential
        membrane_potential = np.zeros(self.num_classes)
        spikes = np.zeros(self.num_classes)

        for t in range(self.simulation_time):
            membrane_potential = self.leaky_factor * membrane_potential + inputs[t]
            firing_neurons = np.where(membrane_potential >= self.threshold)[0]
            membrane_potential[firing_neurons] = 0.0
            spikes[firing_neurons] = 1.0

        return spikes

    def train(self, inputs, targets, num_iterations):
        self.initialize_weights()
        self.initialize_mask()

        for iteration in range(num_iterations):
            outputs = np.zeros_like(targets)
            total_loss = 0.0

            for i in range(inputs.shape[0]):
                x = inputs[i]
                y = targets[i]

                integrated_inputs = np.sum(self.weight_matrix * x.flatten(), axis=1)
                spikes = self.euler_integration(integrated_inputs)
                outputs[i] = spikes

                loss = self.crossentropy_loss(spikes, y)
                total_loss += loss

            # Update the weight matrix based on the evolutionary mask
            if (iteration + 1) % self.Titer == 0:
                self.prune_connections()
                self.regenerate_connections()
                self.weight_matrix = self.mask_matrix * self.weight_matrix

            # Calculate average loss using the modified TET loss function
            average_loss = total_loss / self.simulation_time
            print(f"Iteration {iteration + 1} - Loss: {average_loss}")

            total_elements = self.weight_matrix.size
            non_zero_elements = np.count_nonzero(self.weight_matrix)
            density = non_zero_elements / total_elements
            print(density)


    def prune_connections(self):
        num_connections = self.weight_matrix.size
        num_pruned = int(self.alpha * num_connections)
        indices = np.random.choice(num_connections, num_pruned, replace=False)
        self.mask_matrix.flat[indices] = 0

    def regenerate_connections(self):
        num_connections = self.weight_matrix.size
        num_regenerated = int(self.alpha * num_connections)
        indices = np.random.choice(num_connections, num_regenerated, replace=False)
        self.mask_matrix.flat[indices] = 1

    def crossentropy_loss(self, outputs, targets):
        epsilon = 1e-8
        clipped_outputs = np.clip(outputs, epsilon, 1.0 - epsilon)
        loss = -np.mean(targets * np.log(clipped_outputs))
        return loss

    def predict(self, inputs):
        outputs = np.zeros((inputs.shape[0], self.num_classes))

        for i in range(inputs.shape[0]):
            x = inputs[i]
            integrated_inputs = np.sum(self.weight_matrix * x.flatten(), axis=1)
            spikes = self.euler_integration(integrated_inputs)
            outputs[i] = spikes

        return outputs

    def predict_single_image(self, model, image):
    # Reshape the image to match the input shape expected by the model
      image = image.reshape(1, *model.input_shape)

      # Perform the prediction on the single image
      prediction = model.predict(image)

      # Return the prediction
      return prediction[0]

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Preprocess the data
num_classes = 10
input_shape = (28, 28, 1)  # MNIST images are grayscale with shape (28, 28)
X_train = np.expand_dims(X_train, axis=-1) / 255.0
X_test = np.expand_dims(X_test, axis=-1) / 255.0
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# Training configuration
simulation_time = 10  # Length of simulation time
num_iterations = 5  # Number of training iterations
leaky_factor=0.6
Titer = 1 # Number of iterations for mask update
alpha = 0.5

# Reshape the data
X_train = X_train.reshape(-1, *input_shape)
X_test = X_test.reshape(-1, *input_shape)

# Define and train the Convolutional ESL-SNN model
model = ConvolutionalESLSNNs(input_shape, num_classes, simulation_time, leaky_factor, Titer, alpha)
model.train(X_train, Y_train, num_iterations)

image_to_predict = X_train[0]

# Make a prediction on the single image
prediction = model.predict_single_image(model, image_to_predict)
print(prediction)
plt.imshow(image_to_predict, cmap='gray')
plt.axis('off')  # Remove axis ticks and labels
plt.show()
