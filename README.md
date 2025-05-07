# Face Recognition with Linear Regression

This project is a simple face recognition system using linear regression. It classifies grayscale images into one of three categories: **Elon Musk**, **Cristiano Ronaldo**, or **Unknown**.

## 🧠 How It Works

- Images are converted to grayscale and resized to 50x50 pixels.
- The pixel data is flattened into a 1D array and normalized.
- A linear regression model is trained to classify images into three labels.
- Predictions are based on the closest rounded value (0, 1, or 2), mapped to label names.

## 🚀 Getting Started
- You do not need and additional libraries other than numpy which you can download and install by command:
```bash
pip install numpy
```
- Other libraries are automatically installed with python3.

## Enjoy! 🎈