# Real-Time Diabetic Foot Ulcer Detection

A deep learning-based project for **real-time detection of diabetic foot ulcers (DFUs)** using **thermogram images**. This repository contains training code, evaluation scripts, and deployment-ready models.

---

##  Problem Statement

Diabetic foot ulcers, if undiagnosed, can lead to severe complications and amputations. Manual diagnosis is often time-consuming and subjective. This project develops a **deep learning model** capable of detecting DFUs from thermogram images with high accuracy and deploys it for real-time use.

---

##  Objectives

* Detect diabetic vs. non-diabetic foot thermograms.
* Use **MobileNetV2 with fine-tuning, L2 regularization, and dropout** for robust performance.
* Ensure the model is lightweight and optimized for **mobile deployment**.
* Achieve high accuracy while minimizing inference time and memory usage.

---

##  Dataset

* Input images are provided via **CSV splits**:

  * `train_data.csv`
  * `val_data.csv`
  * `test_data.csv`
* **Classes:** Diabetic, Non-Diabetic.
* **Preprocessing:**

  * Images resized to **224×224**.
  * Normalized using `preprocess_input` from MobileNetV2.
  * Data Augmentation: rotation, horizontal flipping.

---

##  Model Architecture

* **Base Model:** MobileNetV2 pretrained on ImageNet.
* **Fine-tuning:** Last 30 layers unfrozen.
* **Custom Layers:**

  * Global Average Pooling
  * Dense(64, ReLU, L2 regularization)
  * Dropout(0.6)
  * Dense(2, Softmax)

---

##  Training Configuration

* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam (lr = 3e-5)
* **Batch Size:** 32
* **Epochs:** 50
* **Callbacks:**

  * EarlyStopping (patience = 10)
  * ReduceLROnPlateau (factor = 0.1)
  * LearningRateScheduler (decay after 10 epochs)

---

##  Results

* Accuracy and loss curves are plotted during training.
* Final **test accuracy** is printed after evaluation.
* **Output Model:** Saved as `dfu_mobilenetv2.h5`.

---

##  Deployment

1. Convert trained model to **TensorFlow Lite (TFLite)** for Android app integration.
2. Mobile workflow:

   * Capture thermogram image.
   * Preprocess & resize to 224×224.
   * Run inference with TFLite model.
   * Display result (Diabetic / Non-Diabetic).

---

##  Tech Stack

* **Languages:** Python, Java/Kotlin (for Android)
* **Frameworks:** TensorFlow, Keras
* **Deployment Tools:** TensorFlow Lite, Android Studio
* **Libraries:** NumPy, Pandas, OpenCV, Matplotlib, scikit-learn

---

##  Usage

### Clone Repository

```bash
git clone https://github.com/yourusername/diabetic-foot-ulcer-detection.git
cd diabetic-foot-ulcer-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Evaluate the Model

```bash
python evaluate.py
```

### Convert to TFLite

```bash
python convert_to_tflite.py
```

---

##  Future Enhancements

* Expand dataset for improved generalization.
* Add **explainable AI** (Grad-CAM heatmaps).
* Cloud-based storage and monitoring.
* iOS deployment via CoreML.

---

## 👨‍💻 Author

**Soumili Ghosal**
Final Year B.Tech CSE | Deep Learning & Mobile Deployment Enthusiast
