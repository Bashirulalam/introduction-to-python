Fashion-MNIST Classification

📌 Project Overview

This project focuses on building and evaluating machine learning and deep learning models for the Fashion-MNIST dataset, which contains 70,000 grayscale images of fashion items (10 classes such as T-shirt, trouser, bag, shoe, etc.).
The goal is to classify the images into their respective categories using different approaches.

🛠️ Libraries Used

NumPy – Numerical computations

Pandas – Data handling

Matplotlib – Data visualization

Scikit-learn – Machine learning algorithms (Logistic Regression, SVM, etc.)

TensorFlow / Keras – Deep learning models (ANN, CNN)

⚙️ Methods Implemented

Exploratory Data Analysis (EDA)

Visualized sample images

Checked class distribution

Created heatmaps and bar plots

Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

Deep Learning Models

Convolutional Neural Network (CNN) with multiple layers (Conv2D, MaxPooling, Dense, Dropout)

Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📊 Results

Logistic Regression: Moderate performance, limited ability to capture image patterns

SVM: Better than logistic regression, but computationally expensive on large datasets

ANN: Improved accuracy over classical ML models

CNN: Achieved the highest accuracy (around 90%+), demonstrating strong performance in image classification

🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook fashion_mnist.ipynb

📌 Future Improvements

Try more advanced architectures (ResNet, EfficientNet)

Use data augmentation for better generalization

Hyperparameter tuning (learning rate, batch size, optimizer)
