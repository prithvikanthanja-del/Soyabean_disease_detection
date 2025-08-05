import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def is_leaf_image(img: Image.Image) -> bool:
    """
    Basic heuristic check to determine if image resembles a leaf.
    It looks for dominant green color (common in leaves).
    """
    img = img.resize((224, 224))  # Resize to match model input
    img_np = np.array(img)

    # Convert to float32 for color analysis
    img_np = img_np.astype(np.float32)

    # Count pixels where Green is significantly higher than Red and Blue
    green_pixels = np.sum((img_np[:,:,1] > img_np[:,:,0] + 20) & (img_np[:,:,1] > img_np[:,:,2] + 20))
    total_pixels = img_np.shape[0] * img_np.shape[1]

    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.2  # If more than 20% pixels are greenish, assume leaf


# --- Config ---
MODEL_PATH = "model/soya_model.h5"
IMAGE_SIZE = (224, 224)

# Your class labels in the same order as your folders:
CLASS_NAMES = [
    'Caterpillar',
    'Frogeye leaf spot',
    'Healthy',
    'Mosaic',
    'Rust',
    'Septoria brown spot'
]

# --- Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Soybean Disease Detection", page_icon="🌱")
st.title("🌱 Soybean Crop Health Detection")
st.markdown("Upload a soybean leaf or crop image to detect diseases or pest attacks.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    if not is_leaf_image(image):
        st.error("❌ Leaf image not detected. Please upload a valid soybean leaf image.")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Leaf image detected. Processing...")

        # Preprocess
        img_resized = image.resize(IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[predicted_index]

        # Display
        st.markdown(f"### 🧠 Prediction: `{predicted_class}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

        # Chart
        st.subheader("📊 Class Probabilities")
        st.bar_chart({CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))})
else:
    st.info("👈 Upload an image to get started.")
