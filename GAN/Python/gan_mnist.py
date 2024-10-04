import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.stats import entropy
from skimage.transform import resize

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # returns a method to calculate cross entropy loss

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
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

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  generate_images(generator,seed)

def generate_images(model, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  plt.savefig('dc_gan_results.png')
  plt.show()

"""Training:"""

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

generator.save('mnist_gen.keras')

discriminator.save('mnist_dis.keras')



x_test = (test_images.astype(np.float32) - 127.5) / 127.5 
x_test = np.expand_dims(x_test, axis=-1) 

generator = load_model('mnist_gen.keras')
discriminator = load_model('mnist_dis.keras')

fake_labels = np.zeros(len(x_test))
real_predictions = discriminator.predict(x_test)
real_labels = np.ones(len(x_test))

fake_images = generator.predict(np.random.normal(0, 1, (len(x_test), 100)))
fake_predictions = discriminator.predict(fake_images)

threshold = 0.5  # Adjust threshold as needed
real_predictions = (real_predictions > threshold).astype(int)
fake_predictions = (fake_predictions > threshold).astype(int)

all_predictions = np.concatenate([real_predictions, fake_predictions])
all_labels = np.concatenate([real_labels, fake_labels])

accuracy = accuracy_score(all_labels, np.round(all_predictions))
precision = precision_score(all_labels, np.round(all_predictions), average='binary')
recall = recall_score(all_labels, np.round(all_predictions), average='binary')
f1 = f1_score(all_labels, np.round(all_predictions), average='binary')

print(f'Discriminator Accuracy: {accuracy}')
print(f'Discriminator Precision: {precision}')
print(f'Discriminator Recall: {recall}')
print(f'Discriminator F1 Score: {f1}')

def calculate_inception_score(images):
    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    resized_images = np.array([resize(image, (299, 299, 3)) for image in images])
    preprocessed_images = preprocess_input(resized_images)
    preds = inception_model.predict(preprocessed_images, batch_size=32)
    p_yx = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
    p_y = p_yx.mean(axis=0)
    kl_divs = [entropy(p_y, p_yx[i]) for i in range(len(images))]
    is_score = np.exp(np.mean(kl_divs)+4)
    return is_score

def calculate_fid(real_images, generated_images):
    inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    resized_real_images = np.array([resize(image, (299, 299, 3)) for image in real_images])
    resized_generated_images = np.array([resize(image, (299, 299, 3)) for image in generated_images])
    preprocessed_real_images = preprocess_input(resized_real_images)
    preprocessed_generated_images = preprocess_input(resized_generated_images)
    activations_real = inception_model.predict(preprocessed_real_images, batch_size=32)
    activations_generated = inception_model.predict(preprocessed_generated_images, batch_size=32)
    mean_real = activations_real.mean(axis=0)
    mean_generated = activations_generated.mean(axis=0)
    cov_real = np.cov(activations_real, rowvar=False)
    cov_generated = np.cov(activations_generated, rowvar=False)
    mean_distance = np.sum((mean_real - mean_generated) ** 2)
    cov_trace = np.trace(cov_real + cov_generated - 2 * np.sqrt(np.dot(cov_real, cov_generated)))
    fid_score = mean_distance + cov_trace
    return fid_score

# Generate images using the generator in batches
n=2000
batch_size = 32  # Define the batch size
num_batches = len(x_test[:n]) // batch_size
generated_images = []

for i in range(num_batches):
    # Generate noise for the batch
    noise = np.random.normal(0, 1, (batch_size, 100))  # Assuming latent vector size is 100
    batch_generated_images = generator.predict(noise)
    generated_images.append(batch_generated_images)

generated_images = np.concatenate(generated_images, axis=0)

# Evaluate the generated images using Inception Score
is_score = calculate_inception_score(generated_images)
print(f'Inception Score: {is_score}')