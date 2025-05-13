import numpy as np
import pandas as pd
from pytrignomaster import pytrigno
import joblib
import time

# Constants
WINDOW_SIZE = 252
STEP_SIZE = 126
CHANNELS = ['emg1', 'emg3', 'emg4', 'emg5', 'emg6']
LABELS = {0: 'scissors', 1: 'paper', 2: 'rock'}

# Load trained model
model = joblib.load(r"D:\ASU\Term 10\Physiology\biomedical_modellsvmm.pkl")

# Optional: Load scaler if used during training
# scaler = joblib.load(r"D:\ASU\Term 10\Physiology\scaler.pkl")

# Set up Trigno EMG
emg = pytrigno.TrignoEMG(channel_range=(8, 12), samples_per_read=STEP_SIZE)
emg.start()

# Buffer to hold past EMG data
emg_buffer = []

def remove_dc_offset(df):
    return df - df.mean()

def extract_features_from_window(win_df):
    features = []
    for ch in win_df.columns:
        sig = win_df[ch].values
        features.extend([
            np.sqrt(np.mean(sig**2)),              # RMS
            np.mean(np.abs(sig)),                  # Absolute mean
            ((sig[:-1] * sig[1:]) < 0).sum(),      # Zero-crossings
            np.sum(np.abs(np.diff(sig)))           # Waveform length
        ])
    return features

def classify_emg_segment(buffer):
    df = pd.DataFrame(buffer, columns=CHANNELS)
    df = remove_dc_offset(df)
    features = extract_features_from_window(df)

    # Optional: Apply scaler
    # features = scaler.transform([features])[0]

    print("Extracted features:", features)  # Debug: feature values

    prediction = model.predict([features])
    print("Raw prediction:", prediction)    # Debug: raw model output

    return prediction[0]  # Extract scalar label

try:
    print("Listening for EMG data...")
    while True:
        data = emg.read()  # shape (5, 126)
        data = data.T      # shape (126, 5)
        emg_buffer.extend(data.tolist())  # append new samples

        if len(emg_buffer) >= WINDOW_SIZE:
            # Slice the latest window
            current_window = emg_buffer[-WINDOW_SIZE:]
            pred_label = classify_emg_segment(current_window)
            print(f"Detected gesture: {LABELS[pred_label]}")

            # Optional: keep overlapping buffer
            emg_buffer = emg_buffer[-STEP_SIZE:]

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    emg.stop()
