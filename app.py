import os
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

import streamlit as st
import librosa
import librosa.display
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# ===============================
# Grammar Analysis
# ===============================

def analyze_grammar(text):
    import language_tool_python  # lazy import

    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(text)

    return {
        "error_count": len(matches),
        "error_types": list(set(m.ruleId for m in matches))
    }

# ===============================
# Feature Extraction
# ===============================

def extract_audio_features_from_array(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

# ===============================
# Cached ML Resources (SAFE)
# ===============================

@st.cache_resource
def load_resources():
    # 🔑 Lazy imports prevent startup crashes
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small"
    )

    return {"asr": asr}

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
        # Save temporarily ONLY for librosa
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        # Load audio (NO ffmpeg)
        audio, sr = librosa.load(temp_path, sr=16000)

        st.subheader("Audio Analysis")
        st.pyplot(plot_audio_analysis(audio, sr))

        resources = load_resources()

        # ✅ FFMPEG-FREE Whisper call (CRITICAL FIX)
        transcription = resources["asr"](
            {"array": audio, "sampling_rate": sr}
        )["text"]

        st.subheader("Transcription")
        st.code(transcription)

        grammar = analyze_grammar(transcription)

        st.subheader("Grammar Report")
        col1, col2 = st.columns(2)
        col1.metric("Total Errors", grammar["error_count"])
        col2.metric("Unique Error Types", len(grammar["error_types"]))

        # Simple deterministic score (safe fallback)
        score = max(0.0, 5.0 - grammar["error_count"] * 0.2)

        st.subheader("Grammar Score")
        st.metric("Predicted Score", f"{score:.2f} / 5.0")
        st.progress(score / 5.0)

if __name__ == "__main__":
    main()
