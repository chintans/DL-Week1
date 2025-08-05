 # Deep Learning Week 1 - Student Pass/Fail Classification

This repository contains the implementation of a simple perceptron algorithm for classifying students as pass or fail based on their study hours and sleep hours.

## 📁 Project Structure

```
DL-Week1/
├── SampleDataGenerator.ipynb    # Data generation using sklearn
├── Week1_StudentPassFail.ipynb  # Main perceptron implementation
└── README.md                    # This file
```

## 🎯 Project Overview

This project demonstrates the implementation of a **Perceptron** algorithm, one of the fundamental building blocks of neural networks. The goal is to classify students as either "Pass" (1) or "Fail" (0) based on two features:
- **Study Hours**: Number of hours spent studying
- **Sleep Hours**: Number of hours of sleep

## 📊 Dataset

The training data consists of 6 students with the following characteristics:

| Student | Study Hours | Sleep Hours | Outcome |
|---------|-------------|-------------|---------|
| 1       | 1           | 3           | Fail    |
| 2       | 2           | 2           | Fail    |
| 3       | 3           | 6           | Pass    |
| 4       | 4           | 3           | Pass    |
| 5       | 5           | 2           | Pass    |
| 6       | 1           | 6           | Fail    |

## 🧠 Algorithm Implementation

### Perceptron Algorithm
The perceptron is a simple binary classifier that learns a linear decision boundary. The algorithm:

1. **Initialization**: Random weights and bias
2. **Training Loop**: 
   - Forward pass: `z = w₁x₁ + w₂x₂ + b`
   - Activation: `output = 1 if z > 0 else 0`
   - Error calculation: `error = actual - predicted`
   - Weight update: `w = w + learning_rate * error * input`
   - Bias update: `b = b + learning_rate * error`

### Key Components
- **Learning Rate**: 0.1
- **Training Epochs**: 10
- **Activation Function**: Step function (binary threshold)

## 📈 Results

After training, the model learns:
- **Weights**: [1.18470524, -0.4348575]
- **Bias**: -0.7

The decision boundary is visualized showing how the perceptron separates pass and fail students based on their study and sleep patterns.

## 🛠️ Files Description

### `SampleDataGenerator.ipynb`
- Demonstrates data generation using scikit-learn's `make_blobs`
- Creates synthetic 2D data with 2 clusters
- Useful for testing classification algorithms

### `Week1_StudentPassFail.ipynb`
- Main implementation of the perceptron algorithm
- Complete training and visualization pipeline
- Includes:
  - Data preparation
  - Model training
  - Prediction function
  - Decision boundary visualization

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn
```

### Running the Notebooks
1. Open `Week1_StudentPassFail.ipynb` in Jupyter Notebook or Google Colab
2. Run all cells to see the perceptron training process
3. The final cell will display the decision boundary visualization

## 📚 Learning Objectives

This project covers fundamental concepts in deep learning:
- **Linear Classification**: Understanding how linear boundaries separate data
- **Gradient Descent**: Weight updates based on prediction errors
- **Activation Functions**: Step function for binary classification
- **Decision Boundaries**: Visualizing how the model separates classes

## 🔍 Key Insights

The perceptron learns that:
- Students who study more tend to pass (positive weight for study hours)
- Students who sleep more might have mixed effects (negative weight for sleep hours)
- The model finds a linear boundary that best separates pass/fail students

## 🎓 Educational Value

This implementation serves as an excellent introduction to:
- Neural network fundamentals
- Supervised learning concepts
- Binary classification problems
- Linear separability
- Training algorithms

## 📝 Future Enhancements

Potential improvements for this project:
- Implement other activation functions (sigmoid, ReLU)
- Add more features (attendance, previous grades)
- Implement multi-layer perceptron
- Add regularization techniques
- Cross-validation for model evaluation

---

**Note**: This is a simplified educational implementation. Real-world applications would require more sophisticated preprocessing, larger datasets, and advanced neural network architectures.