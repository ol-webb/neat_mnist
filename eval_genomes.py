import neat
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)


x_train_subset = x_train[0:20]
y_train_subset = y_train[0:20]



def eval_genomes(genomes, config):

    inputs = x_train_subset
    labels = y_train_subset

    for genome_id, genome in genomes:
        # Create a neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Initialize fitness score
        fitness = 0

        for xi, label in zip(inputs, labels):
            # Get the network's prediction
            output = net.activate(xi)

            # Find the predicted label (node with the highest activation)
            predicted_label = output.index(max(output))

            # Reward correct predictions
            if predicted_label == label:
                fitness += 1  # Increment fitness for correct prediction

        # Assign fitness to the genome
        genome.fitness = fitness


