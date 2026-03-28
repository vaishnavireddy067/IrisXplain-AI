from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import io
import base64
import csv as csv_module
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__, static_folder="frontend")
CORS(app)

# ── Iris dataset ─────────────────────────────────────────────────
iris = load_iris()
numeric_feature_names = iris.feature_names
iris_target_names = iris.target_names

X = iris.data
y = iris.target
dataset_averages = X.mean(axis=0)
feature_ranges   = X.max(axis=0) - X.min(axis=0)

def _species_title(i: int) -> str:
    return str(iris_target_names[i]).title()

VALIDATION_RANGES = {
    "sepal length (cm)": (0.0, 10.0),
    "sepal width (cm)":  (0.0, 10.0),
    "petal length (cm)": (0.0, 10.0),
    "petal width (cm)":  (0.0, 3.0),
}

# ── Input validation ─────────────────────────────────────────────
def validate_numeric_payload(data: dict):
    if not isinstance(data, dict):
        return None, {"error": "Invalid request body"}
    values, errors = {}, {}
    for feature in numeric_feature_names:
        raw = data.get(feature, "")
        if raw == "" or raw is None:
            errors[feature] = "Missing value"; continue
        try:
            v = float(raw)
        except Exception:
            errors[feature] = "Must be a number"; continue
        min_v, max_v = VALIDATION_RANGES.get(feature, (0.0, 10.0))
        if v <= 0:
            errors[feature] = "Must be a positive number"
        elif v < min_v or v > max_v:
            errors[feature] = f"Must be between {min_v} and {max_v}"
        else:
            values[feature] = v
    if errors:
        return None, {"error": "Invalid input values", "details": errors}
    return values, None

# ── Probability helpers ──────────────────────────────────────────
def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / (e.sum() + 1e-12)

def predict_proba_any(model, x_row):
    x_row = np.asarray(x_row).reshape(1, -1)
    n = len(iris_target_names)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(x_row)[0]
        probs = np.zeros(n)
        for i, c in enumerate(model.classes_):
            if int(c) < n: probs[int(c)] = float(raw[i])
        return probs
    if hasattr(model, "decision_function"):
        sc = model.decision_function(x_row)
        if sc.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-sc[0]))
            probs = np.zeros(n); probs[0] = 1-p1; probs[1] = p1
            return probs
        probs_raw = softmax(sc[0])
        probs = np.zeros(n)
        for i, c in enumerate(model.classes_):
            if int(c) < n: probs[int(c)] = float(probs_raw[i])
        return probs
    pred = int(model.predict(x_row)[0])
    probs = np.zeros(n); probs[pred] = 1.0
    return probs

# ── Sensitivity-based explainability ────────────────────────────
def explain_by_sensitivity(model, x_vec, predicted_idx):
    x_vec = np.asarray(x_vec, dtype=float)
    base_p = predict_proba_any(model, x_vec)[predicted_idx]
    n = len(numeric_feature_names)
    raw = np.zeros(n)
    for i in range(n):
        f = numeric_feature_names[i]
        rng = float(feature_ranges[i]) if feature_ranges[i] > 0 else 1.0
        d = max(0.01, 0.02 * rng)
        min_v, max_v = VALIDATION_RANGES.get(f, (0.0, 10.0))
        xp = x_vec.copy(); xp[i] = min(max_v, xp[i] + d)
        xm = x_vec.copy(); xm[i] = min(max_v, max(1e-6, xm[i] - d))
        raw[i] = abs(predict_proba_any(model, xp)[predicted_idx] - base_p) + \
                 abs(predict_proba_any(model, xm)[predicted_idx] - base_p)
    total = raw.sum()
    imps = (raw / total * 100.0) if total > 1e-12 else np.ones(n) * (100.0 / n)
    ranked = np.argsort(-imps)
    top_factors = [
        {"feature": numeric_feature_names[fi],
         "impact_label": "highest impact" if ri == 0 else "moderately influenced",
         "impact_percent": float(imps[fi])}
        for ri, fi in enumerate(ranked[:2])
    ]
    feat_imp = {numeric_feature_names[i]: float(imps[i]) for i in range(n)}
    vs_avg   = {numeric_feature_names[i]: {"input": float(x_vec[i]),
                                            "average": float(dataset_averages[i]),
                                            "delta": float(x_vec[i] - dataset_averages[i])}
                for i in range(n)}
    return feat_imp, top_factors, vs_avg

def predict_numeric_with_model(model, x_vec):
    probs = predict_proba_any(model, x_vec)
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx]) * 100.0
    fi, tf_, vs = explain_by_sensitivity(model, x_vec, idx)
    return {
        "predicted_class":           _species_title(idx),
        "confidence_percent":        conf,
        "confidence_percent_rounded": int(round(conf)),
        "probabilities":             {str(nm): float(probs[i]) for i, nm in enumerate(iris_target_names)},
        "feature_importance":        fi,
        "top_factors":               tf_,
        "user_input_vs_average":     vs,
    }

# ── Train models ─────────────────────────────────────────────────
def ensure_numeric_models():
    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=250, random_state=42),
        "SVM":                SVC(probability=True, kernel="rbf", random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=2000),
        "Decision Tree":      DecisionTreeClassifier(random_state=42, max_depth=5),
        "KNN":                KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "MLP (Neural Net)":   MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }
    for m in models.values():
        m.fit(X, y)
    return models

print("⏳ Training 6 models ...")
numeric_models = ensure_numeric_models()
print(f"✅ Models ready: {list(numeric_models.keys())}")

_stats_cache = None
def compute_model_stats():
    global _stats_cache
    if _stats_cache:
        return _stats_cache
    stats = {}
    for name, model in numeric_models.items():
        y_pred = model.predict(X)
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        stats[name] = {
            "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
            "precision": round(float(precision_score(y, y_pred, average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y, y_pred, average="macro", zero_division=0)), 4),
            "f1":        round(float(f1_score(y, y_pred, average="macro", zero_division=0)), 4),
            "cv_mean":   round(float(cv.mean()), 4),
            "cv_std":    round(float(cv.std()), 4),
        }
    _stats_cache = stats
    return stats

# ── CNN ──────────────────────────────────────────────────────────
try:
    cnn_model = tf.keras.models.load_model("cnn_model.h5")
    print("✅ CNN model loaded.")
except Exception as e:
    print(f"⚠️  CNN model not loaded: {e}")
    cnn_model = None

try:
    with open("cnn_classes.pkl", "rb") as f:
        cnn_classes = pickle.load(f)
except Exception:
    cnn_classes = ["Class1", "Class2", "Class3"]

def friendly_cnn_label(v):
    s = str(v).replace("_", " ").strip()
    return s.title() if s else "Unknown"

# ── Static files ─────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return app.send_static_file("index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("frontend", path)

# ── API: numeric prediction ──────────────────────────────────────
@app.route("/api/predict_numeric", methods=["POST"])
def predict_numeric():
    if not numeric_models:
        return jsonify({"error": "No models loaded"}), 500
    data = request.json or {}
    features, err = validate_numeric_payload(data)
    if err:
        return jsonify(err), 400
    model_name  = data.get("model_name", "Random Forest")
    compare_all = bool(data.get("compare_all_models", False))
    if compare_all:
        requested = list(numeric_models.keys())
    else:
        if model_name not in numeric_models:
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
        requested = [model_name]
    x_vec = np.array([features[f] for f in numeric_feature_names], dtype=float)
    results = {}
    for name in requested:
        results[name] = predict_numeric_with_model(numeric_models[name], x_vec)
        results[name]["model_name"] = name
    best = max(results, key=lambda k: results[k]["confidence_percent_rounded"])
    avgs = {numeric_feature_names[i]: float(dataset_averages[i]) for i in range(len(numeric_feature_names))}
    comp = [{"model": n, "prediction": results[n]["predicted_class"],
             "confidence_percent_rounded": results[n]["confidence_percent_rounded"]} for n in requested]
    return jsonify({"input": {k: float(v) for k, v in features.items()},
                    "dataset_averages": avgs, "selected_model": model_name,
                    "compare_all_models": compare_all, "results": results,
                    "best_model": best, "best_result": results[best], "comparison_table": comp})

# ── API: image prediction ────────────────────────────────────────
@app.route("/api/predict_image", methods=["POST"])
def predict_image():
    if cnn_model is None:
        return jsonify({"error": "CNN model is not loaded"}), 500
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        raw = request.files["image"].read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((64, 64))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        probs = cnn_model.predict(arr, verbose=0)[0]
        idx   = int(np.argmax(probs))
        pred  = friendly_cnn_label(cnn_classes[idx] if idx < len(cnn_classes) else "Unknown")
        top3  = np.argsort(-probs)[:3]
        top_p = {friendly_cnn_label(cnn_classes[i] if i < len(cnn_classes) else i): float(probs[i]) for i in top3}
        return jsonify({"prediction": pred, "confidence_percent": float(probs[idx]) * 100.0,
                        "confidence_percent_rounded": int(round(float(probs[idx]) * 100.0)),
                        "top_probabilities": top_p})
    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {e}"}), 400

# ── API: image preprocessing preview ────────────────────────────
@app.route("/api/preview_image", methods=["POST"])
def preview_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        raw = request.files["image"].read()
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((64, 64))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"preview_base64": f"data:image/png;base64,{b64}",
                        "size": "64×64 px", "channels": "RGB"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── API: model accuracy scorecard ───────────────────────────────
@app.route("/api/model_stats")
def model_stats():
    try:
        return jsonify(compute_model_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: batch CSV prediction ────────────────────────────────────
@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded"}), 400
    try:
        content = request.files["file"].read().decode("utf-8-sig")
        model_name = request.form.get("model_name", "Random Forest")
        if model_name not in numeric_models:
            model_name = "Random Forest"
        model   = numeric_models[model_name]
        reader  = csv_module.DictReader(io.StringIO(content))
        results = []
        for i, row in enumerate(reader):
            try:
                features, err = validate_numeric_payload(dict(row))
                if err:
                    results.append({"row": i+1, "error": str(err.get("error", "Invalid")),
                                    "predicted_class": "—", "confidence": 0})
                else:
                    x_vec = np.array([features[f] for f in numeric_feature_names], dtype=float)
                    r = predict_numeric_with_model(model, x_vec)
                    results.append({"row": i+1, **{f: float(features[f]) for f in numeric_feature_names},
                                    "predicted_class": r["predicted_class"],
                                    "confidence": r["confidence_percent_rounded"]})
            except Exception as ex:
                results.append({"row": i+1, "error": str(ex), "predicted_class": "—", "confidence": 0})
        return jsonify({"results": results, "count": len(results), "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── API: iris 3D PCA scatter data ───────────────────────────────
@app.route("/api/iris_scatter")
def iris_scatter():
    from sklearn.decomposition import PCA
    pca  = PCA(n_components=3)
    X3d  = pca.fit_transform(X)
    evr  = pca.explained_variance_ratio_.tolist()
    pts  = [{"x": float(X3d[i,0]), "y": float(X3d[i,1]), "z": float(X3d[i,2]),
             "class_id": int(y[i]), "label": str(iris_target_names[y[i]]).title()}
            for i in range(len(y))]
    return jsonify({"points": pts, "variance_explained": evr})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
