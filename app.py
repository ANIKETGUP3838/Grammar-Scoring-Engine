import os
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

import streamlit as st
import librosa
import librosa.display
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import re

# ===============================
# Grammar Analysis (NO JAVA)
# ===============================

def analyze_grammar(text):
    """
    Lightweight grammar heuristic (cloud-safe).
    No Java, no external servers.
    """

    errors = []

    # Repeated words
    errors += re.findall(r"\b(\w+)\s+\1\b", text.lower())

    # Sentence not starting with capital letter
    sentences = re.split(r"[.!?]", text)
    for s in sentences:
        s = s.strip()
        if s and not s[0].isupper():
            errors.append("sentence_case")

    # Missing punctuation at end
    if text and text[-1] not in ".!?":
        errors.append("missing_punctuation")

    return {
        "error_count": len(errors),
        "error_types": list(set(errors))
    }

# ===============================
# Cached ASR (SAFE)
# ===============================

@st.cache_resource
def load_asr():
    from transformers import pipeline  # lazy import
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

# ===============================
# Visualization
# ===============================

def plot_audio_analysis(audio, sr):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(
        D, sr=sr, x_axis="time", y_axis="log", ax=ax[1]
    )
    fig.colorbar(img, ax=ax[1])
    ax[1].set_title("Spectrogram")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax[2])
    fig.colorbar(img, ax=ax[2])
    ax[2].set_title("MFCC")

    return fig

# ===============================
# Main App
# ===============================

def main():
    st.title("🎙️ Advanced Grammar Scoring Engine")

    uploaded_file = st.file_uploader(
        "Upload Audio File", type=["wav", "mp3"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # Load audio (ffmpeg-free)
        audio, sr = librosa.load(temp_path, sr=16000)

        st.subheader("Audio Analysis")
        st.pyplot(plot_audio_analysis(audio, sr))

        asr = load_asr()

        # Whisper transcription (ffmpeg-free)
        transcription = asr(
            {"array": audio, "sampling_rate": sr}
        )["text"]

        st.subheader("Transcription")
        st.code(transcription)

        grammar = analyze_grammar(transcription)

        st.subheader("Grammar Report")
        col1, col2 = st.columns(2)
        col1.metric("Total Errors", grammar["error_count"])
        col2.metric("Unique Error Types", len(grammar["error_types"]))

        # Simple score
        score = max(0.0, 5.0 - grammar["error_count"] * 0.3)

        st.subheader("Grammar Score")
        st.metric("Predicted Score", f"{score:.2f} / 5.0")
        st.progress(score / 5.0)

if __name__ == "__main__":
    main()
