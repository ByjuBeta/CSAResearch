# Image Classifier

This project serves a broad image classifier as a website. Users can upload an image, capture an image from the camera, or draw on a canvas, then the model predicts what the image most likely contains.

This is a more advanced version of the original handwritten digit recognizer. Instead of only recognizing numbers or the 10 CIFAR-10 categories, it uses MobileNetV2 trained on ImageNet, which can recognize about 1,000 real-world image categories.

## Run the Website

```bash
.venv/bin/python app.py
```

Open http://127.0.0.1:8000 in a browser.

## Files

- `app.py` starts the local website and prediction API.
- `image_model.py` loads the pretrained MobileNetV2/ImageNet model and preprocesses images.
- `static/` contains the upload, camera, and drawing website interface.
- `train.py` trains the CIFAR-10 CNN and saves `model/cifar10_classifier.keras`.
- `digit_model.py` and `gui.py` keep the original digit-recognition work available as reference.

## Website Features

- Upload an image from the computer.
- Start the camera and classify a captured frame. The browser may ask for camera permission.
- Draw on a canvas and classify the drawing.
- Show the top prediction and related ImageNet results.

## Retrain the Model

The website now uses a pretrained ImageNet model, so retraining is not required to run the main app. The `train.py` script is still included as a CIFAR-10 training experiment.

```bash
.venv/bin/python train.py
```

The training script saves the model to `model/cifar10_classifier.keras` and the training chart to `model/training_curves.png`.

The first run may need to download the CIFAR-10 dataset. To do a quicker test run, use:

```bash
TRAIN_LIMIT=5000 EPOCHS=2 .venv/bin/python train.py
```

For a better final model, run the full training command without `TRAIN_LIMIT`.
