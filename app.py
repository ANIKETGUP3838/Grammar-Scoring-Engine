import os
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

import streamlit as st
import librosa
import librosa.display
import numpy as np
from transformers import pipeline
import language_tool_python
import tempfile
import matplotlib.pyplot as plt
import tensorflow as tf

# ===============================
# Utility Functions
# ===============================

def analyze_grammar(text):
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)

    error_types = list(set([m.ruleId for m in matches]))

    return {
        "error_count": len(matches),
        "error_types": error_types
    }


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1).reshape(1, -1)


def extract_text_features(text):
    # Simple, safe baseline feature
    return np.array([[len(text.split())]])


# ===============================
# Cached Resources
# ===============================

@st.cache_resource
def load_resources():
    resources = {}

    # Whisper ASR (CPU-safe)
    resources["asr"] = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    # Optional ML model (safe fallback)
    try:
        resources["model"] = tf.keras.models.load_model(
            "models/hybrid_model.h5"
        )
    except Exception:
        resources["model"] = None

    return resources


# ===============================
# Visualization
# ===============================

def plot_audio_analysis(y, sr):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(
        D, sr=sr, x_axis="time", y_axis="log", ax=ax[1]
    )
    fig.colorbar(img, ax=ax[1])
    ax[1].set_title("Spectrogram")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax[2])
    fig.colorbar(img, ax=ax[2])
    ax[2].set_title("MFCC")

    return fig


# ===============================
# Main Application
# ===============================

def main():
    st.title("🎙️ Advanced Grammar Scoring Engine")

    uploaded_file = st.file_uploader(
        "Upload Audio File", type=["wav", "mp3"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())

        # Load audio ONCE
        audio, sr = librosa.load(tmp_file.name, sr=16000)

        st.subheader("Audio Analysis")
        st.pyplot(plot_audio_analysis(audio, sr))

        resources = load_resources()

        # ✅ FIXED: ffmpeg-free Whisper call
        transcription = resources["asr"](
            {"array": audio, "sampling_rate": sr}
        )["text"]

        st.subheader("Transcription")
        st.code(transcription)

        # Grammar Analysis
        grammar_report = analyze_grammar(transcription)

        st.subheader("Grammar Report")
        col1, col2 = st.columns(2)
        col1.metric("Total Errors", grammar_report["error_count"])
        col2.metric(
            "Unique Error Types", len(grammar_report["error_types"])
        )

        # Grammar Score Prediction
        st.subheader("Grammar Score Prediction")

        if resources["model"] is not None:
            audio_features = extract_audio_features(tmp_file.name)
            text_features = extract_text_features(transcription)
            prediction = resources["model"].predict(
                [audio_features, text_features]
            )
            score = float(prediction[0][0])
        else:
            # Safe fallback logic
            score = max(0.0, 5.0 - grammar_report["error_count"] * 0.2)

        st.metric("Predicted Score", f"{score:.2f} / 5.0")
        st.progress(min(score / 5.0, 1.0))


if __name__ == "__main__":
    main()
