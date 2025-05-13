# Rock_Paper_Scissors_Emg_Classifier-
This project implements a machine learning pipeline to classify hand gestures (Rock, Paper, Scissors) using surface EMG signals from six muscle groups. The data is collected using the **Delsys Trigno system** and processed for real-time classification using an **SVM model** and **Pytrigno**.

---

##  Repository Contents

- `Readings.zip`: Contains raw EMG data collected using the Delsys Trigno system from six muscle sensors.
- `model_training.py`: Python script for preprocessing EMG data and training a Support Vector Machine (SVM) classifier to recognize gestures.
- `RealTime.py`: Real-time prediction script using the trained model and Pytrigno to stream live EMG data.
- `pytrignomaster.zip`: Required Pytrigno library for real-time communication with the Trigno system.

---

##  EMG Sensor Placement

EMG signals were collected from the following six muscles:

- Flexor Digitorum Superficialis (FDS)
- Flexor Pollicis Longus (FPL)
- Flexor Digitorum Profundus (FDP)
- Extensor Pollicis Longus (EPL)
- Abductor Pollicis Brevis (APB)
- Extensor Digitorum

These muscles are used to capture distinct activation patterns for:

- **Rock** gesture: Primarily activates FDS and FPL
- **Paper** gesture: Involves FDP, EPL, and APB
- **Scissors** gesture: Recognized based on a combination of differential activity across all channels

---

##  How It Works

1. **Data Collection**  
   Raw EMG signals are recorded via six channels and stored in `Readings.zip`.

2. **Model Training (`model_training.py`)**  
   - Cleans and preprocesses EMG data
   - Extracts features such as RMS, MAV, and waveform length
   - Trains an SVM model to classify the three hand gestures

3. **Real-Time Classification (`RealTime.py`)**  
   - Interfaces with live EMG data using Pytrigno
   - Applies the trained SVM model to predict gestures in real-time

---

##  Requirements

- Python 3.8+
- Packages: `numpy`, `pandas`, `scikit-learn`, `scipy`
- Delsys Trigno System with base station
- Pytrigno (included in this repo)

---
