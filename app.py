
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path
import subprocess
import sys

DIGIT_MODEL_PATH = Path("models/mnist_cnn.keras")
LETTER_MODEL_PATH = Path("models/emnist_letters.keras")

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Character Recognition")
st.write("Recognize handwritten digits and English alphabets using CNN.")

mode = st.selectbox(
    "Choose recognition mode:",
    ["Digits (0–9)", "English Alphabets (A–Z)"]
)

@st.cache_resource
def load_digit_model():
    if not DIGIT_MODEL_PATH.exists():
        with st.spinner("Training digit model..."):
            subprocess.run(
                [sys.executable, "train.py"],
                check=True
            )

    from tensorflow.keras.models import load_model
    return load_model(DIGIT_MODEL_PATH)


@st.cache_resource
def load_letter_model():
    if not LETTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Alphabet model not found. Run python train_letters.py first."
        )

    from tensorflow.keras.models import load_model
    return load_model(LETTER_MODEL_PATH)


try:
    if mode == "Digits (0–9)":
        model = load_digit_model()
        labels = [str(i) for i in range(10)]
        prediction_name = "digit"
    else:
        model = load_letter_model()
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        prediction_name = "letter"

except Exception as exc:
    st.error("The model could not be loaded.")
    st.exception(exc)
    st.stop()


if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

st.caption("Draw one clear character near the center of the box.")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"{mode}_{st.session_state.canvas_key}",
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button(
        f"Predict {prediction_name}",
        type="primary",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "Clear canvas",
        use_container_width=True
    )

if clear:
    st.session_state.canvas_key += 1
    st.rerun()


if predict:
    if (
        canvas.image_data is None
        or np.max(canvas.image_data[:, :, :3]) < 10
    ):
        st.warning("Please draw a character first.")

    else:
        rgba = np.uint8(canvas.image_data)
        rgb = Image.fromarray(rgba).convert("RGB")

        gray = rgb.convert("L").resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        arr = np.asarray(gray, dtype="float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        probabilities = model.predict(arr, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_label = labels[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        if mode == "Digits (0–9)":
            st.success(f"Predicted digit: **{predicted_label}**")
        else:
            st.success(f"Predicted letter: **{predicted_label}**")

        st.write(f"Model confidence: **{confidence:.2f}%**")

        st.subheader("Prediction probabilities")

        st.bar_chart({
            labels[i]: float(probabilities[i])
            for i in range(len(labels))
        })


st.divider()

st.caption(
    "Academic mini-project demonstration | "
    "Datasets: MNIST and EMNIST Letters | "
    "Algorithm: Convolutional Neural Network (CNN)"
)
