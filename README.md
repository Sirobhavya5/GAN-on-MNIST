**Generative Adversarial Networks (GANs) for MNIST Digit Generation**

**Project Overview**

This project explores the implementation and optimization of Generative Adversarial Networks (GANs) to generate realistic hand-written digit images using the MNIST dataset. GANs consist of two neural networks, a generator and a discriminator, trained together in an adversarial framework. The generator learns to create realistic images, while the discriminator distinguishes between real and generated images.
The key focus of this project is on hyperparameter optimization to improve the quality of generated images and enhance the performance of the GAN. Techniques such as grid search and random search were used to find the optimal combination of hyperparameters, including learning rate, batch size, and noise dimension.

**Objectives**

•	Implement a GAN model using TensorFlow and Keras to generate images from the MNIST dataset.
•	Tune hyperparameters (batch size, learning rate, noise dimension) to enhance the quality of generated images.
•	Evaluate model performance using metrics like Inception Score (IS) and Fréchet Inception Distance (FID).
•	Visualize the progress of image generation over training epochs.

**Key Features**

•	Generator and Discriminator Architecture: The generator uses a series of transposed convolutions to upsample noise into images, while the discriminator uses standard convolutions to classify images as real or fake.

•	Hyperparameter Optimization: Hyperparameters like batch size and learning rate were tuned using grid search and random search techniques, significantly improving model convergence and image quality.

•	Evaluation Metrics: In addition to visual inspections, quantitative metrics such as Inception Score (IS) and Fréchet Inception Distance (FID) were used to assess the GAN’s performance.

**Model Architecture**
•	Generator: Consists of dense, batch normalization, and convolutional transpose layers, with Leaky ReLU activation functions to upsample random noise into 28x28 pixel images resembling digits.

•	Discriminator: Composed of convolutional layers with Leaky ReLU activations and dropout to classify real and generated images. The output is a single scalar value representing the probability that the input is a real image.
Hyperparameter Tuning

The following hyperparameters were optimized during the project:

•	Batch Size: Affects training stability and memory usage.

•	Learning Rate: Controls the update step of the optimizer.

•	Noise Dimension: Determines the input vector size for the generator.

Techniques used for optimization:

•	Grid Search: A systematic approach to exploring combinations of hyperparameters.

•	Random Search: A more flexible approach where random combinations of hyperparameters are tested.

**Results**

After tuning the model with 50 epochs and a batch size of 256, we were able to generate high-quality hand-written digits that closely resemble real MNIST digits. The Inception Score and FID were used to quantitatively evaluate the performance, showing significant improvement in image quality as training progressed.

Example of generated digits after training for 50 epochs:

Evaluation Metrics

•	Inception Score (IS): Measures the diversity and quality of generated images.

•	Fréchet Inception Distance (FID): Compares the statistical similarity between real and generated images.
Technologies Used


•	Languages: Python

•	Frameworks: TensorFlow, Keras

•	Dataset: MNIST

•	Evaluation Tools: Inception Score (IS), Fréchet Inception Distance (FID)

•	Development Environment: Google Colab (T4 GPU)

**Conclusion**
This project highlights the importance of hyperparameter optimization in training GANs. By fine-tuning parameters like batch size and learning rate, we achieved substantial improvements in the quality of generated digits. GANs are a powerful tool for generating synthetic data, and this project demonstrates their application on the MNIST dataset.

