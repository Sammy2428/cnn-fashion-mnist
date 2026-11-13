# CNN Image Classification on Fashion-MNIST

End-to-end Convolutional Neural Network (CNN) implementation built with TensorFlow and Keras to classify grayscale apparel images from the **Fashion-MNIST** dataset.

## Dataset Overview
* **Source:** `tf.keras.datasets.fashion_mnist`
* **Size:** 70,000 grayscale images ($28 \times 28 \times 1$), split into 48,000 training, 12,000 validation, and 10,000 test samples.
* **Classes (10):** `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, and `Ankle boot`.

## Model Architecture & Training
* **Feature Extractor:** 3 Convolutional blocks (`Conv2D` with 32, 64, and 128 filters of size $3 \times 3$, `ReLU` activation) paired with two $2 \times 2$ `MaxPooling2D` layers.
* **Classifier Head:** `Flatten` $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(10, softmax)`.
* **Optimization:** Trained for 10 epochs (batch size 64) using the Adam optimizer and Sparse Categorical Cross-Entropy loss.

## Evaluation Results
* **Test Accuracy:** ~91.0% across 10,000 unseen test images
* **Deliverable:** Full code, training/validation curves, sample predictions, and a $10 \times 10$ confusion matrix are available in [`26071DB213.ipynb`](./26071DB213.ipynb).
