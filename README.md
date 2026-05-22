# Handwritten Digit Recognizer

This project trains a convolutional neural network on the MNIST handwritten digit dataset and serves it as a website. Users can draw a digit on the page or upload an image, then the model predicts which number it sees.

## Run the Website

```bash
.venv/bin/python app.py
```

Open http://127.0.0.1:8000 in a browser.

## Files

- `app.py` starts the local website and prediction API.
- `digit_model.py` loads the saved TensorFlow model and preprocesses images.
- `static/` contains the website interface.
- `train.py` trains the MNIST CNN and saves `model/mnist_cnn.h5`.
- `gui.py` keeps the original desktop Tkinter app available.

## Retrain the Model

```bash
.venv/bin/python train.py
```

The training script saves the model to `model/mnist_cnn.h5` and the training chart to `model/training_curves.png`.
