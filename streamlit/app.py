import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Non Accident", "Accident"]
MODEL_PATH = "best_crash_detection_efficientnetb0.keras"
ALERT_THRESHOLD = 0.80  # matches the notebook's email-alert cutoff


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    probs = model.predict(image_array, verbose=0)[0]
    predicted_label = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_label]
    confidence = float(probs[predicted_label])
    return predicted_class, confidence


st.set_page_config(page_title="Accident Detection", page_icon="🚗")
st.title("🚗 CCTV Accident Detection")
st.write("Upload a CCTV frame and the model will classify it as **Accident** or **Non Accident**.")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded frame", use_container_width=True)

    with st.spinner("Running inference..."):
        predicted_class, confidence = predict(model, image)

    if predicted_class == "Accident":
        st.error(f"⚠️ **{predicted_class}** — {confidence:.1%} confidence")
    else:
        st.success(f"✅ **{predicted_class}** — {confidence:.1%} confidence")

    if predicted_class == "Accident" and confidence >= ALERT_THRESHOLD:
        st.warning(f"This would trigger an alert (confidence ≥ {ALERT_THRESHOLD:.0%}).")

    st.progress(confidence)
else:
    st.info("Waiting for an image upload.")
