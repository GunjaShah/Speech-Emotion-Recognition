import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import tempfile
import time

# Load the pre-trained models, label encoder, and scaler
MODELS = {
    'SVM': 'saved_models/svm_model.pkl',
    'Random Forest': 'saved_models/random_forest_model.pkl',
    'KNN': 'saved_models/knn_model.pkl',
    'Logistic Regression': 'saved_models/logistic_regression_model.pkl',
    'Gradient Boosting': 'saved_models/gradient_boosting_model.pkl'
}

SCALER_PATH = 'saved_models/scaler.pkl'
LABEL_ENCODER_PATH = 'saved_models/label_encoder.pkl'

scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 2.5
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
MAX_PAD_LEN = 40

# Logo in Sidebar
logo_path = r"C:\Users\HP\OneDrive\Desktop\Speech Emotion Detection\Speech_Emotion_Recognition_app\Assets\SOU_Logo.png" 
st.sidebar.image(logo_path, use_column_width=True)

def extract_features(audio_data, sr=SAMPLE_RATE, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC, n_fft=512):
    """
    Extracts MFCC features from audio data.
    """
    try:
        with st.spinner("Extracting features..."):
            time.sleep(1)  # Simulating processing time
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
            pad_width = max(0, max_pad_len - mfcc.shape[1])
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            mfcc = mfcc[:, :max_pad_len]
            return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def plot_confidence_bar(emotions, confidences):
    """ Plot the bar chart of confidence levels for each emotion """
    with st.spinner("Plotting confidence levels..."):
        plt.figure(figsize=(10, 4))
        sns.barplot(x=emotions, y=confidences)
        plt.title("Emotion Prediction Confidence Levels")
        plt.ylabel("Confidence")
        plt.xlabel("Emotion")
        plt.ylim(0, 1)  # Set the y-axis limits to 0-1 for better visual clarity
        for i, confidence in enumerate(confidences):
            plt.text(i, confidence + 0.02, f"{confidence:.2f}", ha='center')
        plt.tight_layout()
        st.pyplot(plt)

def predict_emotion(audio_data):
    """
    Predict the emotion based on the given audio data and return all probabilities.
    """
    mfcc = extract_features(audio_data)
    if mfcc is not None:
        with st.spinner("Predicting emotion..."):
            time.sleep(2)  # Simulating processing time
            mfcc_flattened = mfcc.flatten()
            features_scaled = scaler.transform([mfcc_flattened])
            probabilities = model.predict_proba(features_scaled)[0]
            max_index = np.argmax(probabilities)
            predicted_emotion = label_encoder.inverse_transform([max_index])[0]
            confidence = probabilities[max_index]
            return predicted_emotion, confidence, probabilities, label_encoder.classes_
    else:
        return None, None, None, None

# Streamlit app
st.title('Speech Emotion Recognition')

# Sidebar options
st.sidebar.subheader("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Load the selected model with buffering
with st.spinner(f"Loading {model_name} model..."):
    time.sleep(1.5)  # Simulating slight delay for model loading
    model = joblib.load(MODELS[model_name])

# --- File upload section ---
uploaded_files = st.file_uploader("Upload audio file(s)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        st.audio(uploaded_file)

        # Inform the user that the file is being processed
        st.info(f"Processing your voice from file: {uploaded_file.name}")

        # Load the audio file and extract features with buffering
        with st.spinner(f"Processing {uploaded_file.name}..."):
            time.sleep(1)  # Simulating processing time
            audio_data, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, duration=DURATION)
            predicted_emotion, confidence, probabilities, emotion_classes = predict_emotion(audio_data)

        if predicted_emotion and confidence >= confidence_threshold:
            st.success(f"Here is your output! Predicted Emotion for {uploaded_file.name}: {predicted_emotion}")
            st.write(f"Confidence: {confidence:.4f}")

            # Add results for batch processing
            results.append({
                'File Name': uploaded_file.name,
                'Predicted Emotion': predicted_emotion,
                'Confidence': confidence
            })

            # Plot confidence bar
            plot_confidence_bar(emotion_classes, probabilities)
        else:
            st.warning(f"Confidence ({confidence:.4f}) below threshold for {uploaded_file.name}. Skipping prediction.")

    # Export predictions
    if results:
        df_results = pd.DataFrame(results)
        st.subheader("Batch Prediction Results")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="emotion_predictions.csv", mime='text/csv')

# --- Real-time microphone input section ---
st.sidebar.subheader("Real-Time Speech Emotion Recognition")

if st.sidebar.button("Start Recording"):
    st.sidebar.write("Recording... Speak now. 🎤")
    st.info("🎙️ Recording in progress...")

    # Record audio from the microphone
    duration = DURATION  # Record for the same duration as the uploaded file processing
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait for the recording to finish

    st.info("🔊 Stopped recording. Processing your voice...")

    # Save the audio to a temporary file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_wav.name, SAMPLE_RATE, audio_data)  # Save the audio data as a .wav file

    # Load the recorded audio and predict emotion with buffering
    with st.spinner(f"Analyzing microphone input..."):
        audio_data, sr = librosa.load(temp_wav.name, sr=SAMPLE_RATE, duration=DURATION)
        predicted_emotion, confidence, probabilities, emotion_classes = predict_emotion(audio_data)

    if predicted_emotion and confidence >= confidence_threshold:
        st.success(f"Here is your output! 🎧 Predicted Emotion from Microphone Input: {predicted_emotion}")
        st.write(f"Confidence: {confidence:.4f}")

        # Plot confidence bar
        plot_confidence_bar(emotion_classes, probabilities)
    else:
        st.warning("Could not predict the emotion with sufficient confidence.")

    temp_wav.close()

# Adding project contributors' names at the bottom of the sidebar
st.sidebar.markdown("**Project Contributors**")
st.sidebar.markdown("""
- Gunja Shah
- Janvi Bhagchandani
- Snigdha Joshi
- Rajeev Joshi
""")
