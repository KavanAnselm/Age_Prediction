# Age Prediction using ResNet-34 and PyTorch

This project implements a deep learning model to predict a person's age from a facial image. It uses a ResNet-34 model, pre-trained on ImageNet, and fine-tunes it for the regression task of age estimation. The entire pipeline, from data loading and augmentation to training and detailed evaluation, is built using the PyTorch framework.

## Key Features

- **Model:** Fine-tuned ResNet-34 architecture for regression.
- **Framework:** PyTorch
- **Task:** Age Estimation (Regression) from images.
- **Loss Function:** Mean Absolute Error (L1 Loss).
- **Optimizer:** AdamW.
- **Techniques:** Transfer learning, data augmentation (random horizontal flip, color jitter), and detailed performance analysis.

## Model Performance

The model was trained for 20 epochs and evaluated on a test set of 47,568 images. The key performance metric is the Mean Absolute Error (MAE), which represents the average error in years.

#### Overall Metrics

| Metric | Value | 95% Confidence Interval |
| :--- | :--- |:---|
| **MAE** | **5.837 years** | `[5.785, 5.893]` |
| **RMSE** | 8.408 years | `[8.311, 8.506]` |
| **R² Score** | 0.671 | - |
| **Bias** | -1.563 years | - |
| **Accuracy (±5 years)** | 57.6% | `[57.2%, 58.1%]` |

#### MAE by Age Group

The model's performance varies across different age ranges, showing higher accuracy for younger individuals.

| Age Bucket | MAE (years) |
| :--- | :--- |
| 1–5 | **1.31** |
| 6–12 | **5.98** |
| 13–18 | **6.56** |
| 19–30 | **4.23** |
| 31–45 | **5.38** |
| 46–60 | **7.96** |
| 61–100 | **10.85** |

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/age-prediction-pytorch.git](https://github.com/your-username/age-prediction-pytorch.git)
    cd age-prediction-pytorch
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Requirements

The project was developed using Python 3.11. The necessary packages are listed below.

```
# requirements.txt
torch>=1.13
torchvision>=0.14
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
Pillow>=9.4
tqdm>=4.64
```

---

## Usage

### 1. Data Preparation

This model requires a dataset with a specific directory structure. The root data directory should contain `train` and `test` subfolders. Inside each, there should be folders named after the age, containing the corresponding images.

```
<DATA_DIR>/
├── train/
│   ├── 001/
│   │   ├── image_1.jpg
│   │   └── image_2.png
│   ├── 002/
│   │   └── ...
│   └── ...
└── test/
    ├── 001/
    │   ├── image_100.jpg
    │   └── ...
    └── ...
```

### 2. Training the Model

1.  Open the `Age_Prediction_Model Ver-4.ipynb` notebook.
2.  In the constants cell (the 3rd code cell), update the `DATA_DIR` variable to point to your dataset's root directory.
3.  Run all cells in the notebook to train the model from scratch. The trained model weights will be saved as `age_prediction_resnet34.pth`.

### 3. Inference on a Single Image

To predict the age from a single image, you can use the `predict_age` function defined in the notebook. Ensure the trained model file `age_prediction_resnet34.pth` is in the same directory.

```python
# Example from the notebook
from PIL import Image
import torch
from torchvision import models, transforms

# Load the trained model
model = models.resnet34(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load('age_prediction_resnet34.pth'))
model.to(DEVICE)
model.eval()

# Use the prediction function
image_path = "path/to/your/image.jpg"
predicted_age_years, predicted_age_months = predict_age(image_path, model, DEVICE)
```

---

## Hardware (GPU and CPU Use)

-   **GPU:** The code is configured to automatically use a CUDA-enabled GPU if one is detected. Training on a GPU is **highly recommended** for performance. The model was trained and tested on a system with CUDA version 11.8.
-   **CPU:** If a GPU is not available, the code will default to using the CPU for both training and inference. Note that training on a CPU will be significantly slower.
