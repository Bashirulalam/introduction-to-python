👗 Fashion-MNIST Classification
📌 Project Overview

This project focuses on building and evaluating machine learning and deep learning models for the Fashion-MNIST dataset, which contains 70,000 grayscale images of clothing items across 10 classes (e.g., T-shirt, trouser, bag, shoe, etc.).
The goal is to classify images into their respective categories using different approaches.

🛠️ Libraries Used

🧮 NumPy – Numerical computations

🗂️ Pandas – Data handling

📊 Matplotlib – Data visualization

🤖 Scikit-learn – ML algorithms (Logistic Regression, SVM, etc.)

🧠 TensorFlow / Keras – Deep learning models (ANN, CNN)

⚙️ Methods Implemented
🔍 Exploratory Data Analysis (EDA)

Visualized sample images

Checked class distribution

Created heatmaps and bar plots

🤖 Machine Learning Models

Logistic Regression

Support Vector Machine (SVM)

🧠 Deep Learning Models

Artificial Neural Network (ANN)

Convolutional Neural Network (CNN):

Conv2D

MaxPooling

Dense

Dropout

📏 Evaluation Metrics

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📊 Results
Model	Accuracy	Notes
Logistic Regression	~75%	Limited ability to capture image features
SVM	~82%	Better than LR, but slow on large datasets
ANN	~86%	Outperforms classical ML models
CNN	90%+	Best results, strong performance in image classification
🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/fashion-mnist-classification.git
cd fashion-mnist-classification


Install dependencies:

pip install -r requirements.txt


Run the notebook:

jupyter notebook fashion_mnist.ipynb

📌 Future Improvements

⚡ Try more advanced architectures (ResNet, EfficientNet)

🔄 Apply data augmentation for better generalization

🎯 Perform hyperparameter tuning (learning rate, batch size, optimizer)
