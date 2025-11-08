# ECG Arrhythmia Classification using Hybrid CNN-BiLSTM
A deep learning project to classify cardiac arrhythmias from single-lead ECG signals. This project addresses significant class imbalance in the MIT-BIH Arrhythmia Dataset by utilizing a hybrid Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) architecture, enhanced with class-weighted training.
## Overview
Electrocardiogram (ECG) monitoring is standard for diagnosing cardiac irregularities. Manual interpretation is time-consuming and prone to error. This project automates the classification of individual heartbeats into 5 standard categories defined by the AAMI (Association for the Advancement of Medical Instrumentation).

**Key Features:**
* **Imbalance Handling:** Utilizes computed class weights during training to prevent bias toward the majority "Normal" class.
* **Hybrid Architecture:** Combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal pattern recognition.
* **Robust Evaluation:** Analyzes performance using Confusion Matrices, F1-scores per class, and ROC-AUC curves.

## Dataset
Source: [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

### Description:

The dataset is pre-processed into individual heartbeats.
* **Input:** 187 time-step vector (representing a single heartbeat).
* **Classes:** 5 categories (mapped from original annotations).

### Dataset Structure:

**Files:**
- **mitbih_train.csv:** ~87,000 samples  
- **mitbih_test.csv:** ~21,000 samples  

**Features:**
- **187 features:** ECG signal values (time-series data points)  
- **1 target column:** Classification labels (classes 0–4)

**Classes:**

The dataset contains five types of heartbeat classifications:

| **Class** | **Type** | **Description** |
|:----------:|:----------|:----------------|
| 0 | Normal (N) | Normal heartbeat |
| 1 | Supraventricular (S) | Supraventricular premature beat |
| 2 | Ventricular (V) | Ventricular premature beat |
| 3 | Fusion (F) | Fusion of ventricular and normal beat |
| 4 | Unclassifiable (Q) | Unclassifiable beat |


## Model Architecture
A hybrid CNN-BiLSTM approach designed to capture both the shape of a single beat and the sequence dependencies.

1.  **CNN Block (Spatial Feature Extraction):** 3 layers of 1D Convolutions (filters: 32, 64, 128) with Batch Normalization, ReLU activation, and Max Pooling to extract morphological features (like QRS complex shape).
2.  **BiLSTM Block (Temporal Feature Learning):** 2 layers of Bidirectional LSTMs (64 units) to capture temporal dependencies in both forward and backward directions.
3.  **Dense Block (Classification Head):** 2 Fully Connected layers with Dropout (0.5) for final 5-class Softmax prediction.

| **Layers** | **Filters** | **Components** |
|--------------------|-----------------------|----------------|
| **Input** | (187, 1) | Raw ECG signal input |
| **CNN Block 1** | 32 filters | Conv1D → BatchNorm → ReLU → MaxPooling(2) → Dropout(0.2) |
| **CNN Block 2** | 64 filters | Conv1D → BatchNorm → ReLU → MaxPooling(2) → Dropout(0.2) |
| **CNN Block 3** | 128 filters | Conv1D → BatchNorm → ReLU → MaxPooling(2) → Dropout(0.2) |
| **Bi-LSTM 1** | 64 units | Bidirectional LSTM → Dropout(0.3) |
| **Bi-LSTM 2** | 64 units | Bidirectional LSTM |
| **Dense Layer** | 64 units | Dense(64, activation='relu') → Dropout(0.5) |
| **Output Layer** | 5 classes | Dense(5, activation='softmax') |

### Key Techniques:
- **Batch Normalization:** Stabilizes and accelerates model training by normalizing activations across mini-batches.  
- **Progressive Dropout:** Gradually increases dropout rates *(0.2 → 0.3 → 0.5)* across layers to reduce overfitting.  
- **Class Weights:** Balances the influence of each heartbeat class to handle class imbalance in ECG data.  
- **Bidirectional LSTM:** Captures temporal dependencies in both forward and backward directions for ECG sequences.  
- **Early Stopping:** Prevents overfitting by halting training when validation loss stops improving.  
- **Learning Rate Reduction:** Dynamically lowers the learning rate when validation accuracy plateaus for smoother convergence.  

### Design Rationale:
- **CNNs for Spatial Features:** Convolutional layers effectively detect local morphological patterns in ECG signals, such as PQRST wave shapes and abnormal spikes.  
- **LSTMs for Temporal Dynamics:** Recurrent layers capture long-term dependencies and rhythmic variations across consecutive heartbeats.  
- **Hybrid Strength:** Combining CNN and LSTM allows the model to learn both spatial (waveform structure) and temporal (beat sequence) representations.  
- **Bidirectional Context:** Bidirectional LSTMs enhance temporal awareness by processing sequences in both forward and backward directions, leading to a deeper contextual understanding of arrhythmia patterns.
- **Training Stability:** Batch Normalization accelerates convergence and prevents internal covariate shifts, ensuring stable learning across epochs.  
- **Robust Generalization:** Progressive Dropout (0.2 → 0.3 → 0.5) and Class Weighting help the model generalize better and remain resilient to class imbalance in ECG datasets.

## Results
The model achieved an overall accuracy of **~97%** on the test set. Due to class imbalance, F1-score and AUC are better indicators of performance.

* **Test Accuracy:** 96.91%
* **Test Loss:** 0.1014
* **Micro-Averaged AUC:** 0.9984

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal (0)** | 1.00 | 0.97 | 0.98 | 18118 |
| **Supraventricular (1)**| 0.58 | 0.88 | 0.70 | 556 |
| **Ventricular (2)** | 0.94 | 0.96 | 0.95 | 1448 |
| **Fusion (3)** | 0.54 | 0.90 | 0.67 | 162 |
| **Unclassifiable (4)** | 0.98 | 0.99 | 0.99 | 1608 |

## Installation & Usage

### Prerequisites

- Python 3.10+
- Any system with TensorFlow support

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/madhav2905/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

2. **Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```
*(Ensure you have `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, and `plotly` installed)*

4. **Run Notebooks:**
* Start with `notebooks/EDA.ipynb` to explore the data and generate preprocessed `.npy` files.
* Run `notebooks/ModelTraining.ipynb` to train the CNN-BiLSTM model and evaluate results.

## Contributions
Contributions are welcome! Areas for improvement:
* Data augmentation techniques
* Ensemble methods
* Attention mechanisms
* Real-time inference optimization
* Mobile deployment

## License
This project is open source and available under the MIT License.

## Acknowledgments
* MIT-BIH Arrhythmia Laboratory for the dataset.
* Kaggle community for dataset preprocessing and insights.

## Contact
For questions or collaborations, feel free to open an issue or reach out!
