import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


scissors = r'C:/Users/Catherine/Downloads/biomedicalcato/scissors.csv'
paper = r'C:/Users/Catherine/Downloads/biomedicalcato/paper.csv'
rock = r'C:/Users/Catherine/Downloads/biomedicalcato/rock.csv'



# extract EMG columns for sensors 1 to 6 (kolohom zay ba3d)
def extract_emg_signals(df):
    return pd.DataFrame({
        'emg1': df.iloc[:, 1],   # Column B
        'emg2': df.iloc[:, 15],  # Column P
        'emg3': df.iloc[:, 29],  # Column AD
        'emg4': df.iloc[:, 43],  # Column AR
        'emg5': df.iloc[:, 57],  # Column BF
        'emg6': df.iloc[:, 71],  # Column BT
    })
#skip the first 7 rows
scissors_df = pd.read_csv(scissors, skiprows=6)
paper_df = pd.read_csv(paper, skiprows=6)
rock_df = pd.read_csv(rock, skiprows=6)

emg_scissors = extract_emg_signals(scissors_df)
emg_paper = extract_emg_signals(paper_df)
emg_rock = extract_emg_signals(rock_df)

emg_scissors['label'] = 'scissors'
emg_paper['label'] = 'paper'
emg_rock['label'] = 'rock'

emg_all = pd.concat([emg_scissors, emg_paper, emg_rock], ignore_index=True)
emg_all.head()

def clean_emg_data(emg_df):
    cleaned = emg_df.copy()
    for i in range(1, 7):
        cleaned[f'emg{i}'] = pd.to_numeric(cleaned[f'emg{i}'], errors='coerce')  # Coerce to NaN
        cleaned[f'emg{i}'] = cleaned[f'emg{i}'].fillna(0)  # Replace NaNs with 0 (or use .interpolate())
    return cleaned  # ← This was missing

# clean data before plotting
emg_scissors = clean_emg_data(emg_scissors)
emg_paper = clean_emg_data(emg_paper)
emg_rock = clean_emg_data(emg_rock)



def plot_emg_signals(emg_df, title):
    plt.figure(figsize=(15, 8))
    for i in range(1, 7):
        plt.subplot(3, 2, i)
        plt.plot(emg_df[f'emg{i}'], linewidth=0.7)
        plt.title(f'EMG{i} - {title}')
        plt.xlabel('Sample')
        plt.ylabel('mV')
    plt.tight_layout()
    plt.show()

plot_emg_signals(emg_scissors, 'Raw - Scissors')

def remove_dc_offset(emg_df):
    dc_removed = emg_df.copy()
    for i in range(1, 7):
        dc_removed[f'emg{i}'] = dc_removed[f'emg{i}'] - dc_removed[f'emg{i}'].mean()
    return dc_removed

emg_scissors_clean = remove_dc_offset(emg_scissors)
emg_paper_clean = remove_dc_offset(emg_paper)
emg_rock_clean = remove_dc_offset(emg_rock)

plot_emg_signals(emg_scissors_clean, 'DC Removed - Scissors')

for i in range(1, 7):
    mean_before = emg_scissors[f'emg{i}'].mean()
    mean_after = emg_scissors_clean[f'emg{i}'].mean()
    print(f'EMG{i} - Mean before: {mean_before:.4f}, after: {mean_after:.4f}')


t= scissors_df.iloc[:, 0]


#compute differences
dts = np.diff(t)

#average Δt
mean_dt = np.mean(dts)

#sampling rate
fs = 1.0 / mean_dt

print(f"Mean delta t = {mean_dt:.7f} s")
print(f"Sampling rate ≈ {fs:.1f} Hz")

"""Window size=200ms×1.26kHz=252samples

Step size=100ms×1.26kHz=126samples
"""

def calculate_rms_per_channel(emg_df, window_size=252, step_size=126):
    rms_values = {f'emg{i}': [] for i in range(1, 7)}

    for start in range(0, len(emg_df) - window_size + 1, step_size):
        window = emg_df.iloc[start:start + window_size]
        for i in range(1, 7):
            signal = window[f'emg{i}'].values
            rms = np.sqrt(np.mean(signal ** 2))
            rms_values[f'emg{i}'].append(rms)

    return pd.DataFrame(rms_values)

# Calculate and plot
rms_df_s = calculate_rms_per_channel(emg_scissors_clean)
rms_df_p = calculate_rms_per_channel(emg_paper_clean)
rms_df_r = calculate_rms_per_channel(emg_rock_clean)


# Plot all channels
rms_df_s.plot(figsize=(12, 5), title='scissors RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

rms_df_p.plot(figsize=(12, 5), title='paper RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

rms_df_r.plot(figsize=(12, 5), title='rock RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

# drop emg2 from the rms dataframes
rms_df_s_no_emg2 = rms_df_s.drop(columns=['emg2'])
rms_df_p_no_emg2 = rms_df_p.drop(columns=['emg2'])
rms_df_r_no_emg2 = rms_df_r.drop(columns=['emg2'])

# recalculate overall rms without emg2
rms_total_s = np.sqrt((rms_df_s_no_emg2**2).sum(axis=1) / 5)
rms_total_p = np.sqrt((rms_df_p_no_emg2**2).sum(axis=1) / 5)
rms_total_r = np.sqrt((rms_df_r_no_emg2**2).sum(axis=1) / 5)

# Plot
plt.plot(rms_total_s, label='Scissors')
plt.plot(rms_total_p, label='Paper')
plt.plot(rms_total_r, label='Rock')
plt.title('Total RMS per Window (All Channels Except EMG2)')
plt.xlabel('Window Index')
plt.ylabel('Total RMS')
plt.legend()
plt.grid(True)
plt.show()

def calculate_rms_per_channel(emg_df, window_size=252, step_size=126):
    rms_values = {f'emg{i}': [] for i in [1, 3, 4, 5, 6]}  # Excluding emg2

    for start in range(0, len(emg_df) - window_size + 1, step_size):
        window = emg_df.iloc[start:start + window_size]
        for i in [1, 3, 4, 5, 6]:  # Excluding emg2
            signal = window[f'emg{i}'].values
            rms = np.sqrt(np.mean(signal ** 2))
            rms_values[f'emg{i}'].append(rms)

    return pd.DataFrame(rms_values)
# Calculate and plot
rms_df_s = calculate_rms_per_channel(emg_scissors_clean)
rms_df_p = calculate_rms_per_channel(emg_paper_clean)
rms_df_r = calculate_rms_per_channel(emg_rock_clean)


# Plot all channels
rms_df_s.plot(figsize=(12, 5), title='scissors RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

rms_df_p.plot(figsize=(12, 5), title='paper RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

rms_df_r.plot(figsize=(12, 5), title='rock RMS per EMG channel')
plt.xlabel('Window Index')
plt.ylabel('RMS')
plt.grid(True)
plt.show()

def calculate_total_rms(emg_df, window_size=252, step_size=126):
    rms_values = []

    for start in range(0, len(emg_df) - window_size + 1, step_size):
        window = emg_df.iloc[start:start + window_size]
        total_energy = np.sqrt(np.mean(window[['emg1', 'emg3', 'emg4', 'emg5', 'emg6']].values ** 2))  # excluding emg2
        rms_values.append(total_energy)

    return np.array(rms_values)

# only the channels we care about
CHANNELS = ['emg1','emg3','emg4','emg5','emg6']

def segment_and_extract_no_emg2(df, label,
                                window_size=252,
                                step_size=126,
                                rms_thresh=0.05):
    """
    df           : DataFrame with columns emg1…emg6 (but we'll skip emg2)
    label        : integer label for this gesture (0/1/2)
    window_size  : samples per window
    step_size    : hop length
    rms_thresh   : skip windows with too little activity
    """
    X, y = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        win = df.iloc[start:start+window_size]

        # overall RMS across our five channels
        overall_rms = np.sqrt(np.mean(win[CHANNELS].values**2))
        if overall_rms < rms_thresh:
            continue

        fv = []
        for ch in CHANNELS:
            sig = win[ch].values
            # 1) RMS
            fv.append(np.sqrt(np.mean(sig**2)))
            # 2) Absolute mean
            fv.append(np.mean(np.abs(sig)))
            # 3) Zero-crossing count
            fv.append(((sig[:-1]*sig[1:])<0).sum())
            # 4) Waveform length
            fv.append(np.sum(np.abs(np.diff(sig))))
        X.append(fv)
        y.append(label)

    return np.array(X), np.array(y)

# --- apply to each gesture ---
X_s, y_s = segment_and_extract_no_emg2(emg_scissors_clean, label=0)
X_p, y_p = segment_and_extract_no_emg2(emg_paper_clean,    label=1)
X_r, y_r = segment_and_extract_no_emg2(emg_rock_clean,     label=2)

# --- combine into one dataset ---
X = np.vstack([X_s, X_p, X_r])
y = np.concatenate([y_s, y_p, y_r])

print("Windows per class:", len(y_s), len(y_p), len(y_r))
print("Total windows:", len(y))

"""#plotting time domain features

"""


def segment_and_extract_no_emg2(df, label, window_size=252, step_size=126, rms_thresh=0.05):
    X, y = [], []
    features_by_channel = { 'RMS': [], 'AbsMean': [], 'Waveform': [], 'ZCR': [] }

    for start in range(0, len(df) - window_size + 1, step_size):
        win = df.iloc[start:start+window_size]
        overall_rms = np.sqrt(np.mean(win[CHANNELS].values**2))
        if overall_rms < rms_thresh:
            continue

        rms_vals, mean_vals, wv_vals, zcr_vals = [], [], [], []

        for ch in CHANNELS:
            sig = win[ch].values
            rms_vals.append(np.sqrt(np.mean(sig**2)))
            mean_vals.append(np.mean(np.abs(sig)))
            zcr_vals.append(((sig[:-1]*sig[1:]) < 0).sum())
            wv_vals.append(np.sum(np.abs(np.diff(sig))))

        # Save all features
        X.append(rms_vals + mean_vals + zcr_vals + wv_vals)
        y.append(label)

        # Also collect per-feature lists (averaged per window for plotting)
        features_by_channel['RMS'].append(np.mean(rms_vals))
        features_by_channel['AbsMean'].append(np.mean(mean_vals))
        features_by_channel['ZCR'].append(np.mean(zcr_vals))
        features_by_channel['Waveform'].append(np.mean(wv_vals))

    return np.array(X), np.array(y), features_by_channel

X_s, y_s, feats_s = segment_and_extract_no_emg2(emg_scissors_clean, label=0)
X_p, y_p, feats_p = segment_and_extract_no_emg2(emg_paper_clean,    label=1)
X_r, y_r, feats_r = segment_and_extract_no_emg2(emg_rock_clean,     label=2)

# Combine data
X = np.vstack([X_s, X_p, X_r])
y = np.concatenate([y_s, y_p, y_r])

print("Windows per class:", len(y_s), len(y_p), len(y_r))
print("Total windows:", len(y))


fs = 1259.3
step_size = 126
window_duration = step_size / fs

# Time axes for each gesture
t_s = np.arange(len(feats_s['RMS'])) * window_duration
t_p = np.arange(len(feats_p['RMS'])) * window_duration
t_r = np.arange(len(feats_r['RMS'])) * window_duration

features = ['RMS', 'AbsMean', 'Waveform', 'ZCR']
titles = ['RMS', 'Absolute Mean', 'Waveform Length', 'Zero Crossings']

for feat, title in zip(features, titles):
    plt.figure(figsize=(12, 4))
    plt.plot(t_s, feats_s[feat], label='Scissors', color='blue')
    plt.plot(t_p, feats_p[feat], label='Paper', color='green')
    plt.plot(t_r, feats_r[feat], label='Rock', color='red')
    plt.title(f"{title} over Time for Each Gesture")
    plt.xlabel("Time (s)")
    plt.ylabel(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""#Frequency Domain"""



# Constants
FS = 1259.3  # Sampling frequency
WINDOW_SIZE = 252  # Same as your time domain analysis
GESTURES = ['Scissors', 'Paper', 'Rock']
COLORS = ['blue', 'green', 'red']

def extract_gesture_frequency_features(emg_data, gesture_name):

    all_fft = []
    all_power = []

    # Get the channel names from the dataframe
    channels = [col for col in emg_data.columns if col.startswith('emg')]

    # Process each window
    for start in range(0, len(emg_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = emg_data.iloc[start:start+WINDOW_SIZE]

        # Average across all channels for this window
        avg_signal = window[channels].mean(axis=1).values

        # FFT analysis
        freqs = np.fft.rfftfreq(WINDOW_SIZE, 1/FS)
        fft_values = np.abs(np.fft.rfft(avg_signal))
        power_spectrum = fft_values ** 2

        # Store for aggregation
        all_fft.append(fft_values)
        all_power.append(power_spectrum)

        # CWT analysis (compute for first window only for visualization)
        if start == 0:
            scales = np.arange(1, 128)
            cwt_coeffs, freqs_cwt = pywt.cwt(avg_signal, scales, 'morl', sampling_period=1/FS)
            cwt_power = np.abs(cwt_coeffs)**2

    # Aggregate across windows
    mean_fft = np.mean(all_fft, axis=0)
    mean_power = np.mean(all_power, axis=0)

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # FFT Magnitude
    plt.subplot(2, 2, 1)
    plt.plot(freqs, mean_fft, color='b')
    plt.title(f"{gesture_name} - Average FFT Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Power Spectrum
    plt.subplot(2, 2, 2)
    plt.plot(freqs, 10*np.log10(mean_power), color='r')
    plt.title(f"{gesture_name} - Average Power Spectrum (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid(True)

    # CWT Power Spectrum (first window only)
    plt.subplot(2, 1, 2)
    im = plt.imshow(10*np.log10(cwt_power), aspect='auto',
                   extent=[0, WINDOW_SIZE/FS, freqs_cwt[-1], freqs_cwt[0]],
                   cmap='jet', interpolation='nearest')
    plt.title(f"{gesture_name} - CWT Power Spectrum (First Window)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Power (dB)')

    plt.tight_layout()
    plt.show()

    return {
        'mean_freq': np.sum(freqs * mean_power) / np.sum(mean_power),
        'median_freq': freqs[np.where(np.cumsum(mean_power) >= np.sum(mean_power)/2)[0][0]],
        'total_power': np.sum(mean_power),
        'band_powers': {
            'low': np.sum(mean_power[(freqs >= 20) & (freqs < 50)]),
            'mid': np.sum(mean_power[(freqs >= 50) & (freqs < 150)]),
            'high': np.sum(mean_power[(freqs >= 150) & (freqs < 500)])
        }
    }

# Dictionary to store all features
all_gesture_features = {}

# Analyze each gesture
print("Frequency Domain Feature Extraction Results:")
for emg_data, gesture in zip([emg_scissors_clean, emg_paper_clean, emg_rock_clean], GESTURES):
    features = extract_gesture_frequency_features(emg_data, gesture)
    all_gesture_features[gesture] = features

    # Print feature summary
    print(f"\n{gesture}:")
    print(f"Mean Frequency: {features['mean_freq']:.2f} Hz")
    print(f"Median Frequency: {features['median_freq']:.2f} Hz")
    print(f"Total Power: {features['total_power']:.2e}")
    print("Band Power Ratios:")
    total = features['total_power']
    for band, power in features['band_powers'].items():
        print(f"  {band}: {power/total:.2%}")

# Comparative Visualization
plt.figure(figsize=(12, 6))

# Plot mean frequency comparison
plt.subplot(1, 2, 1)
for i, gesture in enumerate(GESTURES):
    plt.bar(i, all_gesture_features[gesture]['mean_freq'], color=COLORS[i], label=gesture)
plt.title("Mean Frequency Comparison")
plt.xticks(range(len(GESTURES)), GESTURES)
plt.ylabel("Frequency (Hz)")
plt.legend()

# Plot band power ratios comparison
plt.subplot(1, 2, 2)
width = 0.25
for i, band in enumerate(['low', 'mid', 'high']):
    for j, gesture in enumerate(GESTURES):
        ratio = all_gesture_features[gesture]['band_powers'][band]/all_gesture_features[gesture]['total_power']
        plt.bar(j + i*width, ratio, width=width, color=COLORS[j],
                label=gesture if i == 0 else "")
plt.xticks(np.arange(len(GESTURES)) + width, GESTURES)
plt.title("Band Power Distribution")
plt.ylabel("Power Ratio")
plt.legend()

plt.tight_layout()
plt.show()

"""#Model"""



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)

clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['scissors','paper','rock']))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Get feature importances
importances = clf.feature_importances_
num_channels = 5
feature_names = []

# We have 4 features per channel: RMS, AbsMean, ZCR, Waveform
base_features = ['RMS', 'AbsMean', 'ZCR', 'Waveform']
for ch in range(num_channels):
    for feat in base_features:
        feature_names.append(f"{feat}_ch{ch+1}")

# Plot
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45, ha="right")
plt.tight_layout()
plt.grid(True)
plt.show()


# Support Vector Machine
svm_clf = SVC(kernel='rbf', C=1, gamma='scale')  # You can change kernel to 'linear', 'poly', etc.
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_acc * 100:.2f}%")


joblib.dump(svm_clf, r"D:\ASU\Term 10\Physiology\biomedical_modellsvm.pkl")
joblib.dump(clf, r"D:\ASU\Term 10\Physiology\biomedical_model.pkl")