import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
import joblib

warnings.filterwarnings('ignore')

# ----------------------------- Configuration ----------------------------- #

# Replace this with your actual dataset path
DATASET_PATH = r'C:\Users\HP\OneDrive\Desktop\Speech Emotion Recognition\TESS Toronto emotional speech set data'  # Update this path

# Audio parameters
SAMPLE_RATE = 22050  # Sampling rate in Hz
DURATION = 2.5       # Duration of each audio clip in seconds
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)  # Ensure integer

# Feature extraction parameters
N_MFCC = 40          # Number of MFCCs to extract
MAX_PAD_LEN = 40     # Maximum padding length for MFCCs

# ----------------------------- Feature Extraction ----------------------------- #

def extract_features(file_path, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from an audio file.

    Parameters:
    - file_path (str): Path to the audio file.
    - max_pad_len (int): Maximum length for padding/truncating MFCCs.
    - n_mfcc (int): Number of MFCCs to extract.

    Returns:
    - np.ndarray: Flattened MFCC feature vector.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        # Ensure SAMPLES_PER_TRACK is integer
        if len(audio) < SAMPLES_PER_TRACK:
            padding = int(SAMPLES_PER_TRACK - len(audio))  # Cast to int
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > SAMPLES_PER_TRACK:
            audio = audio[:SAMPLES_PER_TRACK]  # Truncate if longer
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Pad or truncate MFCC to have the same shape
        if mfcc.shape[1] < max_pad_len:
            pad_width = int(max_pad_len - mfcc.shape[1])  # Cast to int
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ----------------------------- Label Extraction ----------------------------- #

def extract_label(folder_name):
    """
    Extracts the emotion and actress from the folder name to form a unique class label.
    Example:
        'OAF_angry' -> 'oaf_angry'
        'OAF_Pleasant_surprise' -> 'oaf_pleasant_surprise'
    """
    parts = folder_name.split('_')
    if len(parts) >= 2:
        # Join all parts after the first underscore
        emotion_part = '_'.join(parts[1:])
        return f"{parts[0].lower()}_{emotion_part.lower()}"
    else:
        # If no underscore is found, use the entire folder name as label
        print(f"Warning: Folder name '{folder_name}' does not contain an underscore. Using full name as label.")
        return folder_name.lower().strip()

# ----------------------------- Data Loading ----------------------------- #

def load_data(dataset_path):
    """
    Loads data from the dataset directory, extracts features, and assigns labels.

    Parameters:
    - dataset_path (str): Path to the dataset directory.

    Returns:
    - X (np.ndarray): Feature array.
    - y (np.ndarray): Label array.
    """
    features = []
    labels = []
    emotion_folders = os.listdir(dataset_path)
    print(f"Found {len(emotion_folders)} folders in the dataset.")

    for folder in emotion_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder_path}, not a directory.")
            continue
        label = extract_label(folder)
        file_count = 0
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(label)
                    file_count += 1
        print(f"Processed {file_count} files in folder '{folder}'.")

    X = np.array(features, dtype='float32')  # Optimize data type
    y = np.array(labels)

    # Verify sample counts per class
    unique, counts = np.unique(y, return_counts=True)
    print("\nSample counts per class:")
    for label, count in zip(unique, counts):
        print(f"{label}: {count}")

    return X, y

# ----------------------------- Model Training and Evaluation ----------------------------- #

def train_and_evaluate(X, y):
    """
    Trains multiple classifiers and evaluates their performance.

    Parameters:
    - X (np.ndarray): Feature array.
    - y (np.ndarray): Label array.

    Returns:
    - results (dict): F1 scores for each model.
    - cm_dict (dict): Confusion matrices for each model.
    - le (LabelEncoder): Fitted label encoder.
    - X_test (np.ndarray): Test feature array.
    - y_test (np.ndarray): Test label array.
    - models (dict): Trained models.
    - scaler (StandardScaler): Fitted scaler.
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Verify label distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    label_counts = dict(zip(le.inverse_transform(unique), counts))
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Check if any class has fewer than 2 samples
    insufficient_classes = [label for label, count in label_counts.items() if count < 2]
    if insufficient_classes:
        print("\nError: The following classes have fewer than 2 samples, which is insufficient for stratification:")
        for label in insufficient_classes:
            print(f"- {label}")
        raise ValueError("Some classes have fewer than 2 samples. Please check your dataset.")

    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models with optimized hyperparameters
    models = {
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1  # Utilize all available cores
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=50,          # Reduced from 100
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42, 
            verbosity=1,
            tree_method='gpu_hist',   # GPU accelerated algorithm
            gpu_id=0, 
            predictor='gpu_predictor'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=50,          # Reduced from 100
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=1,
            device='gpu',             # Use GPU
            gpu_platform_id=0,
            gpu_device_id=0,
            n_jobs=-1                 # Utilize all available cores
        )
    }

    results = {}
    cm_dict = {}

    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            start_time = time.time()
            if name in ['XGBoost', 'LightGBM']:
                # Models with GPU support and early stopping
                model.fit(
                    X_train, y_train,
                    early_stopping_rounds=10,
                    eval_set=[(X_test, y_test)],
                    verbose=True
                )
            else:
                # Other models
                model.fit(X_train, y_train)
            end_time = time.time()
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = f1
            cm = confusion_matrix(y_test, y_pred)
            cm_dict[name] = cm
            print(f"{name} F1 Score: {f1:.4f}")
            print(f"{name} Training Time: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error training {name}: {e}")

    return results, cm_dict, le, X_test, y_test, models, scaler

# ----------------------------- Visualization ----------------------------- #

def plot_f1_scores(results):
    """
    Plots F1 scores for different models.

    Parameters:
    - results (dict): Dictionary containing F1 scores for each model.
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.ylabel('F1 Score')
    plt.title('Model Comparison based on F1 Score')
    plt.ylim(0, 1)
    for index, value in enumerate(results.values()):
        plt.text(index, value + 0.01, f"{value:.2f}", ha='center', va='bottom')
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plots a confusion matrix.

    Parameters:
    - cm (np.ndarray): Confusion matrix.
    - classes (list): List of class names.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

# ----------------------------- Main Execution ----------------------------- #

def main():
    print(f"Dataset Path: {DATASET_PATH}")  # For verification
    print("Loading data...")
    X, y = load_data(DATASET_PATH)
    print(f"\nData loaded. Feature shape: {X.shape}, Labels shape: {y.shape}")

    # Verify that data is loaded
    if X.shape[0] == 0:
        raise ValueError("No data was loaded. Please check your dataset path and structure.")

    print("\nTraining and evaluating models...")
    try:
        results, cm_dict, label_encoder, X_test, y_test, models, scaler = train_and_evaluate(X, y)
    except ValueError as ve:
        print(f"Error during training and evaluation: {ve}")
        return

    print("\nF1 Scores for all models:")
    for model_name, f1 in results.items():
        print(f"{model_name}: {f1:.4f}")

    # Plot F1 Scores
    plot_f1_scores(results)

    # Plot Confusion Matrix for each model
    class_names = label_encoder.classes_
    for model_name, cm in cm_dict.items():
        plot_confusion_matrix(cm, classes=class_names, title=f'Confusion Matrix for {model_name}')

    # Print Classification Reports
    print("\nClassification Reports:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

    # Optionally, save the models and scaler for future use
    # Example:
    # joblib.dump(models['Random Forest'], 'random_forest_model.pkl')
    # joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    main()


