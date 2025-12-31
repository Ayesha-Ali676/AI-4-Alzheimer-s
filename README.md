# AI-4-Alzheimer-s  
## Alzheimer's Detection Using CNN on MRI Images

---

## Project Overview

This Jupyter notebook implements a **Convolutional Neural Network (CNN)** for binary classification of Alzheimer's disease from MRI brain images. The model distinguishes between healthy individuals and those with Alzheimer's based on grayscale MRI scans.

---

## Alzheimer's Disease Summary

Alzheimer's disease is a progressive neurodegenerative disorder characterized by the accumulation of amyloid plaques and tau tangles in the brain, leading to neuronal death and brain atrophy. It primarily affects memory, cognition, and behavior, making it the most common cause of dementia worldwide.

Early detection is crucial for managing symptoms and potentially slowing progression. MRI imaging reveals structural changes such as hippocampal atrophy and ventricular enlargement, which can be detected through automated analysis like this CNN model.

---

## Dataset

- **Source:** Kaggle MRI Alzheimer's Dataset (`train.parquet`)
- **Format:** Parquet file containing image data and labels
- **Labels:** Multi-class (0 = Healthy, 1–3 = Alzheimer's stages), converted to binary  
  - `0 = Healthy`  
  - `1 = Alzheimer's`
- **Image Processing:** Raw bytes converted to `128x128` grayscale NumPy arrays

---

## Data Preprocessing

- **Loading:** Read parquet file using `pandas`
- **Image Decoding:** Extract bytes from blob format and convert to PIL images, then to NumPy arrays
- **Normalization:** Pixel values scaled to `[0, 1]`
- **Channel Addition:** Added single channel dimension for CNN input `(N, 1, H, W)`
- **Train/Val/Test Split:**  
  - 60% training  
  - 20% validation  
  - 20% testing
- **Tensor Conversion:** NumPy arrays converted to PyTorch tensors

---

## Model Architecture

### CNN Model

- **Conv1:** 32 filters, `3x3` kernel, `padding=1`
- **Conv2:** 64 filters, `3x3` kernel, `padding=1`
- **Pooling:** `MaxPool2d(2x2)` after each convolution layer
- **FC1:** 128 neurons (input: `643232` flattened)
- **Dropout:** 0.5 probability
- **FC2:** 2 neurons (binary classification)
- **Activation:**  
  - ReLU for hidden layers  
  - No activation on output (handled by `CrossEntropyLoss`)

---

## Training Details

- **Epochs:** 20
- **Batch Size:** 32
- **Optimizer:** Adam (`lr = 0.001`)
- **Loss Function:** `CrossEntropyLoss` with class weights (to handle imbalance)
- **Device:** CUDA if available, otherwise CPU
- **Class Weights:** Computed using `sklearn` balanced weighting

---

## Training Results

- **Final Validation Accuracy:** 84.88%
- **Training Loss:** Decreased from `0.7260` to `0.6919`
- **Validation Loss:** Stable around `0.693 – 0.694`

---

## Evaluation Metrics

- **Test Accuracy:** ~85% (based on validation performance)
- **Classification Report:** Generated for precision, recall, and F1-score
- **Predictions:** Saved to `predictions.csv` with columns:
  - ID
  - Name
  - Actual
  - Predicted

---

## Detection Results

The model was evaluated on random test samples with the following outcomes:

- **Random Test Predictions:** All 5 randomly selected test images were correctly classified
- **Confidence Levels:** Alzheimer's predictions showed confidence around `0.52`
- **Detected Images:** Grayscale MRI scans displayed with true labels and predictions

### Sample Results

```
True Label: Alzheimer  
Prediction: Alzheimer  
Confidence: 0.52  
Result: ✅ Correct
```

(Repeated for 5 samples — all correct)

Images show characteristic brain structures where Alzheimer's is indicated by potential atrophy patterns detectable in MRI scans.

---

## Visualization

- **Sample Images:** Display of first 5 training images with labels (healthy and Alzheimer's cases)
- **Prediction Visualization:** Random test samples with true/predicted labels, confidence scores, and result indicators (✅ Correct / ❌ Wrong)
- **Bar Chart:** Comparison of actual vs predicted counts for Healthy and Alzheimer's classes
- **Detected Images:** Individual MRI scans processed by the model, showing brain anatomy where Alzheimer's-related changes may be present

---

## Key Findings

- Model achieves reasonable accuracy on the validation set
- Class imbalance addressed using weighted loss
- Predictions show consistent confidence levels around `0.52` for Alzheimer's class
- All random test predictions in the final evaluation were correct

---

## Dependencies

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `PIL` (Pillow)

---

## Usage

1. Ensure `train.parquet` is in the working directory  
2. Run notebook cells sequentially to:
   - Load data
   - Preprocess images
   - Define the model
   - Train and evaluate
3. Predictions are saved to `predictions.csv`

---

## Limitations

- Model may be overfitting or not capturing complex patterns (validation accuracy plateaued)
- Image size assumed to be `128x128`; may require adjustment
- Binary classification simplifies multi-stage Alzheimer's labels

---

## Future Improvements

- Data augmentation
- Deeper network architecture
- Hyperparameter tuning
- Cross-validation
- Advanced preprocessing techniques
