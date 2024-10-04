import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time
from IPython import display
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Define the generator architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Define the discriminator architecture
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Define loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the Inception Score calculation function
def calculate_inception_score(images):
    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    resized_images = np.array([resize(image, (299, 299, 3)) for image in images])
    preprocessed_images = preprocess_input(resized_images)
    preds = inception_model.predict(preprocessed_images, batch_size=32)
    p_yx = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
    kl_divs = [entropy(p_y, p_yx[i]) for i in range(len(images))]
    is_score = np.exp(np.mean(kl_divs)+4)
    return is_score

# Define training function for the generator and discriminator networks
def train(train_dataset, epochs,bs):
	BATCH_SIZE = bs
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    for epoch in range(epochs):
        start = time.time()
        for image_batch in train_dataset:
            train_step(image_batch, generator, discriminator)
        # Save checkpoints every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    generate_and_evaluate(generator)

# Define training step function
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Define generator loss function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generate and evaluate images using the generator
def generate_and_evaluate(generator):
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    generated_images = generator(seed, training=False)
    is_score = calculate_inception_score(generated_images)
    print(f'Inception Score: {is_score}')
    # Generate and save comparison tables for generator and discriminator
    generate_comparison_tables(generator)

# Generate comparison tables for generator and discriminator networks
def generate_comparison_tables(generator):
    # Generate images using the generator
    num_batches = 5
    batch_size = 32
    generated_images = []
    for i in range(num_batches):
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        batch_generated_images = generator.predict(noise)
        generated_images.append(batch_generated_images)
    generated_images = np.concatenate(generated_images, axis=0)
    # Evaluate discriminator on real images
    real_predictions = discriminator.predict(train_images[:len(generated_images)])
    real_labels = np.ones(len(generated_images))
    # Evaluate discriminator on fake images
    fake_predictions = discriminator.predict(generated_images)
    fake_labels = np.zeros(len(generated_images))
    # Combine real and fake predictions and labels
    all_predictions = np.concatenate([real_predictions, fake_predictions])
    all_labels = np.concatenate([real_labels, fake_labels])
    # Calculate discriminator metrics
    accuracy = accuracy_score(all_labels, np.round(all_predictions))
    precision = precision_score(all_labels, np.round(all_predictions), average='binary')
    recall = recall_score(all_labels, np.round(all_predictions), average='binary')
    f1 = f1_score(all_labels, np.round(all_predictions), average='binary')
    # Create and save discriminator comparison table
    discriminator_comparison_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    discriminator_comparison_table.to_csv('discriminator_comparison_table.csv', index=False)
    # Calculate Inception Score for generated images
    is_score = calculate_inception_score(generated_images)
    # Create and save generator comparison table
    generator_comparison_table = pd.DataFrame({
        'Learning Rate': [generator_optimizer.learning_rate.numpy()],
        'Batch Size': [BATCH_SIZE],
        'Inception Score': [is_score]
    })
    generator_comparison_table.to_csv('generator_comparison_table.csv', index=False)

# Train the GAN

for epochs in [50,80,110]:
	for BatchSize in [32,64,128]:
		train_comp(train_dataset,epochs,BatchSize)
		

train(train_dataset, EPOCHS)
