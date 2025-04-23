import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "dataset"
LABELS = {"real": 0, "ai": 1}

X, y = [], []

# Extract features
for label in LABELS:
    folder = os.path.join(DATASET_PATH, label)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            y_audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            X.append(mfcc_mean)
            y.append(LABELS[label])
        except Exception as e:
            print(f"Failed on {filename}: {e}")

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SVC(kernel="linear", probability=True)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "voice_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model trained and saved.")