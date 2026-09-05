import os
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# SETTINGS
# =========================================================

st.set_page_config(
    page_title="Accident Detection",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Non Accident",
    "Accident"
]

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_crash_detection_efficientnetb0.keras"
)


# =========================================================
# BACKEND
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize exactly like training
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert image to numpy array
    image_array = np.array(
        image
    )

    # EfficientNet preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict_image(model, image):

    # Preprocess image
    image_array = preprocess_image(
        image
    )

    # Prediction
    probs = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get predicted class
    predicted_label = int(
        np.argmax(probs)
    )

    # Get class name
    predicted_class = CLASS_NAMES[
        predicted_label
    ]

    # Get confidence
    confidence = float(
        probs[predicted_label]
    )

    return predicted_class, confidence, probs


# =========================================================
# FRONTEND - CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #0f172a,
            #111827
        );
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .model-badge {
        text-align: center;
        margin: 15px auto 30px auto;
        padding: 10px 20px;
        width: fit-content;
        border-radius: 20px;
        background-color: #1f2937;
        color: #d1d5db;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 40px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🚨 Accident Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered accident detection using EfficientNetB0'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="model-badge">'
    'EfficientNetB0 · 224×224 · 2 Classes'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Failed to load the model."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# UPLOAD IMAGE
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# DISPLAY IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔍 Detect Accident",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            predicted_class, confidence, probs = predict_image(
                model,
                image
            )

        confidence_percent = (
            confidence * 100
        )


        # =================================================
        # RESULT
        # =================================================

        if predicted_class == "Accident":

            st.error(
                f"🚨 Accident Detected\n\n"
                f"Confidence: {confidence_percent:.2f}%"
            )

            st.progress(
                confidence
            )

        else:

            st.success(
                f"✅ No Accident\n\n"
                f"Confidence: {confidence_percent:.2f}%"
            )

            st.progress(
                confidence
            )


        # =================================================
        # PROBABILITIES
        # =================================================

        st.write("### Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Non Accident",
                f"{probs[0] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Accident",
                f"{probs[1] * 100:.2f}%"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ACCIDENT DETECTION · EFFICIENTNETB0 · AI PROJECT
    </div>
    """,
    unsafe_allow_html=True
)

