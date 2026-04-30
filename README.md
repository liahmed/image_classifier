# CIFAR-10 Image Classification with Custom CNN

Complete end-to-end image classification pipeline using TensorFlow/Keras and a custom CNN trained on CIFAR-10.

## Project Structure

```text
image_classifier/
│
├── data/
├── output/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── saved_model/
│   └── cifar10_model.h5
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   └── evaluator.py
├── train.py
├── predict.py
├── web_app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── config.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train.py
```

What training does:
- Downloads CIFAR-10 automatically.
- Applies normalization and augmentation.
- Trains a custom CNN with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
- Saves:
  - best model to `saved_model/cifar10_model.h5`
  - training curves to `output/training_curves.png`
  - confusion matrix to `output/confusion_matrix.png`
  - classification report to `output/classification_report.txt`

Expected test accuracy after full training is around **~85%** (can vary by seed/system).

## Predict on a Single Image

```bash
python predict.py --image path/to/image.jpg
```

Example output:

```text
Top-3 predictions:
1. automobile — 91.3%
2. truck — 6.2%
3. ship — 1.8%
```

## Frontend (Detection + Segmentation)

Run:

```bash
python web_app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

What it does:
- Uploads your image from the browser.
- Detects objects with a pretrained segmentation model.
- Draws bounding boxes, class labels, and segmentation masks.
- Shows detected object labels with confidence scores.

## Sample Output Screenshots

Add screenshots here after running training:
- `output/training_curves.png`
- `output/confusion_matrix.png`
- terminal prediction output screenshot

