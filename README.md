# 🌸 IrisXplain AI

> **Smart Classification using Advanced Machine Learning**

A full-stack ML web application that classifies Iris flower species using 6 scikit-learn models and performs image classification using a CNN — with explainability, multi-model comparison, and an interactive AI-powered dashboard.

---

## 📸 Preview

| Landing Page | Dashboard |
|---|---|
| Animated splash screen with branding | Full prediction & explainability UI |

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌿 **6 ML Models** | Random Forest, SVM, Logistic Regression, Decision Tree, KNN, MLP (Neural Net) |
| 📊 **Explainable AI (XAI)** | Sensitivity-based feature importance per prediction |
| 🖼️ **CNN Image Classifier** | Upload any image for deep learning classification |
| 📈 **Model Scorecard** | View live global test accuracy, F1-scores, and cross-validation stats |
| 🌌 **3D PCA Scatter Plot** | Visualize the entire Iris dataset interactively in 3D feature space |
| 🗃️ **Batch CSV Prediction** | Upload a CSV file to get bulk predictions and confidence scores |
| 🧠 **Ask AI** | Rule-based conversational explanation of predictions |
| 🎤 **Voice Fill** | Speak your 4 Iris measurements to auto-fill the form |
| 📄 **PDF Report** | Download a summary report of your predictions |
| 🕘 **Prediction History** | Last 5 predictions saved in browser localStorage |
| 🌸 **Animated Landing Page** | Beautiful intro screen with particles and transitions |

---

## 🗂️ Project Structure

```
ml_project/
├── ml.py                  # Flask backend — main entry point
├── requirements.txt       # Python dependencies
├── cnn_model.h5           # Pre-trained CNN model (TensorFlow/Keras)
├── cnn_classes.pkl        # Class labels for the CNN
├── rf_model.pkl           # (unused at runtime — models retrained fresh)
├── svm_model.pkl
├── lr_model.pkl
├── dataset/               # Training data (if any)
└── frontend/
    ├── index.html         # Landing page + main app shell
    ├── app.js             # All frontend logic (prediction, charts, history)
    ├── style.css          # Dark-theme design system
    └── images/
        ├── setosa.png
        ├── versicolor.png
        └── virginica.png
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone / Download the Project

```bash
cd ml_project
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow installation may take a few minutes.

### 4. Run the App

```bash
python ml.py
```

The server starts at **http://localhost:5000**

---

## 🌐 Usage

1. Open **http://localhost:5000** in your browser.
2. The **IrisXplain AI** landing page appears with animations.
3. Click **🚀 Launch Dashboard** to enter the app.
4. Fill in the **Iris measurements** (sepal/petal length & width).
5. Choose a model and click **⚡ Predict**.
6. View the predicted species, confidence %, feature importance chart, and explainability panel.
7. Toggle **Compare All Models** to see a full multi-model breakdown.
8. Optionally upload an image in the **Image Classification** section.
9. Type a question in **Ask AI** (e.g., *"why"*, *"confidence"*, *"feature"*) and click **🤖 Ask AI**.
10. Click **📄 Download Report (PDF)** to export your results.

---

## 🔌 API Endpoints

### `POST /api/predict_numeric`

Classify an Iris flower from numeric measurements.

**Request Body (JSON):**
```json
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2,
  "model_name": "Random Forest",
  "compare_all_models": false
}
```

**Response:**
```json
{
  "best_model": "Random Forest",
  "best_result": {
    "predicted_class": "Setosa",
    "confidence_percent": 100.0,
    "feature_importance": { ... },
    "top_factors": [ ... ],
    "probabilities": { ... }
  },
  "comparison_table": [ ... ],
  "dataset_averages": { ... }
}
```

---

### `POST /api/predict_image`

Classify an uploaded image using the CNN model.

**Request:** `multipart/form-data` with field `image` (PNG or JPEG).

**Response:**
```json
{
  "prediction": "Class Name",
  "confidence_percent": 87.4,
  "confidence_percent_rounded": 87,
  "top_probabilities": { ... }
}
```

---

## 🤖 ML Models

### Numeric (Iris Dataset)

All numeric models are **retrained fresh** at startup on the Iris dataset to avoid pickle version compatibility issues.

| Model | Type | Notes |
|---|---|---|
| Random Forest | Ensemble | 250 estimators |
| SVM | Kernel-based | RBF kernel, probability=True |
| Logistic Regression | Linear | max_iter=2000 |
| Decision Tree | Tree | max_depth=5 |
| KNN | Instance-based | k=7, distance weights |
| MLP (Neural Net) | Deep Learning | 2 hidden layers (64, 32 neurons) |

### Image (CNN)

- Loaded from `cnn_model.h5` (TensorFlow/Keras)
- Input images resized to **64×64** pixels
- RGBA images automatically converted to RGB
- Class labels loaded from `cnn_classes.pkl`

---

## 🧠 Explainability (XAI)

IrisXplain uses **sensitivity-based perturbation** to explain predictions:

1. For each feature, slightly increase and decrease the value.
2. Measure the change in the predicted class probability.
3. Features with the highest total impact are ranked as most influential.
4. Top-2 factors are displayed as human-readable bullet points.

This approach works across **all model types** (trees, SVMs, linear, KNN) without requiring model-specific introspection.

---

## 📦 Dependencies

```
flask
flask-cors
numpy
pandas
scikit-learn
tensorflow
Pillow
opencv-python
matplotlib
```

---

## 🎨 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, scikit-learn, TensorFlow |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **Charts** | Chart.js v4 |
| **PDF Export** | jsPDF |
| **Fonts** | Google Fonts — Inter, Space Grotesk |
| **Design** | Custom dark-theme CSS with glassmorphism |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| Port 5000 already in use | Change port: `PORT=5001 python ml.py` |
| CNN model not loading | Ensure `cnn_model.h5` exists in the project root |
| TensorFlow install fails | Try `pip install tensorflow-cpu` instead |
| Voice Fill not working | Requires Chrome/Edge and microphone permission |

---

## 📄 License

This project is for educational and demonstration purposes.

---

<div align="center">
  <strong>🌸 IrisXplain AI</strong> — Built with Flask · scikit-learn · TensorFlow · Vanilla JS
</div>
