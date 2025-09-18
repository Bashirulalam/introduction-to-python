👗 Fashion-MNIST Classification

📌 Project Overview

This project focuses on building and evaluating machine learning and deep learning models for the Fashion-MNIST dataset, which contains 70,000 grayscale images of clothing items across 10 classes (e.g., T-shirt, trouser, bag, shoe, etc.).
The goal is to classify images into their respective categories using different approaches.

🛠️ Libraries Used

1.  NumPy – Numerical computations
2.  Pandas – Data handling
3.  Matplotlib – Data visualization
4.  Scikit-learn – ML algorithms (Logistic Regression, SVM, etc.)
5.  TensorFlow / Keras – Deep learning models (ANN, CNN)

⚙️ Methods Implemented

1. Exploratory Data Analysis (EDA)
2. Visualized sample images
3. Checked class distribution
4. Created heatmaps and bar plots

🤖 Machine Learning Models

1. Logistic Regression
2. Support Vector Machine (SVM)
3.  Convolutional Neural Network (CNN):

📏 Evaluation Metrics

1. Accuracy
2. Confusion Matrix
3. Classification Report (Precision, Recall, F1-score)

📊 Results
Model	Accuracy	Notes
1. Logistic Regression	84%	Limited ability to capture image features
2. SVM	88%	Better than LR, but slow on large datasets
3. CNN	90%	Best results, strong performance in image classification


📌 Future Improvements

1. Try more advanced architectures (ResNet, EfficientNet)
2. Apply data augmentation for better generalization
3. Perform hyperparameter tuning (learning rate, batch size, optimizer)
