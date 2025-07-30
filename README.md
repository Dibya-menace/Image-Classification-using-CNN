# 📷 Image Classification using CNN (TensorFlow & Keras)

This project involves building a deep learning-based **image classification model** using **Convolutional Neural Networks (CNNs)** to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The model is developed using **Python**, **TensorFlow**, and **Keras**.

---

## 🚀 Project Overview

* **Goal**: Classify input images into one of the predefined categories (e.g., airplane, automobile, bird, cat, etc.).
* **Dataset**: CIFAR-10 — consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
* **Tools & Libraries**: Python, TensorFlow, Keras, NumPy, Matplotlib, scikit-learn.

---

## 🧠 Model Highlights

* Utilized a **CNN architecture** with multiple convolutional and pooling layers followed by dense layers for classification.
* Employed **data preprocessing techniques** such as:

  * Normalization of pixel values.
  * Data augmentation (e.g., rotation, flipping, zoom).
* Applied **dropout** to prevent overfitting.
* Trained and validated the model on labeled data.
* Evaluated the model using:

  * **Accuracy score**
  * **Confusion matrix**
  * **Real-time predictions** on unseen test images.

---

## 📁 File Structure

```
├── Image_classification.ipynb  # Main Jupyter notebook with training & evaluation
├── README.md                   # Project description and instructions
└── (Optional folders for datasets, saved models, outputs)
```

---

## 🧪 Results

* Achieved high accuracy on the CIFAR-10 test dataset.
* Real-time predictions demonstrate the model's generalization capabilities.
* Confusion matrix used to analyze misclassifications per class.

---

## 🛠 How to Run

1. **Install dependencies**:

```bash
pip install tensorflow matplotlib scikit-learn
```

2. **Run the notebook**:

Open `Image_classification.ipynb` using Jupyter or any compatible IDE.

3. **Dataset**:

The dataset is automatically downloaded via Keras:

```python
from tensorflow.keras.datasets import cifar10
```

---

## 📊 Example Predictions

Images from the test set are displayed alongside the model's predicted and true labels for visual evaluation.

---

## 📌 Key Learnings

* Gained hands-on experience with CNNs and image data preprocessing.
* Understood challenges in multiclass classification and how to handle them.
* Applied machine learning metrics for rigorous model evaluation.

---

## 📚 References

* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
* [Keras Tutorials](https://keras.io/examples/)

