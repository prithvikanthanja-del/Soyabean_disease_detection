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
    return green_ratio > 0.08 # If more than 20% pixels are greenish, assume leaf


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
DISEASE_INFO = {
    "Caterpillar": {
        "description": "Caterpillars feed on soybean leaves, reducing photosynthesis and plant health.",
        "treatment": "Use organic insecticides like neem oil or introduce natural predators."
    },
    "Frogeye leaf spot": {
        "description": "A fungal disease causing circular lesions with dark borders and gray centers.",
        "treatment": "Apply fungicides early and rotate crops to prevent spread."
    },
    "Healthy": {
        "description": "This leaf appears healthy and free from visible diseases or pests.",
        "treatment": "No treatment needed. Continue regular monitoring."
    },
    "Mosaic": {
        "description": "Soybean mosaic virus causes leaf wrinkling and mottling with light/dark green patterns.",
        "treatment": "Use virus-free seeds and control aphid populations."
    },
    "Rust": {
        "description": "Caused by Phakopsora pachyrhizi fungus; shows as orange/reddish pustules.",
        "treatment": "Use rust-resistant varieties and apply systemic fungicides."
    },
    "Septoria brown spot": {
        "description": "Starts as small brown spots on lower leaves, may defoliate plants.",
        "treatment": "Use crop rotation and apply preventive fungicides at early stages."
    }
}


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
        st.warning("⚠️ This may not look like a typical green soybean leaf. Proceeding anyway...")

    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Leaf image detected.")

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

        # 📘 Disease Info
        info = DISEASE_INFO.get(predicted_class)
        if info:
            st.subheader("📘 Disease Information")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Treatment:** {info['treatment']}")

        # Chart
        st.subheader("📊 Class Probabilities")
        st.bar_chart({CLASS_NAMES[i]: float(prediction[i]) for i in range(len(CLASS_NAMES))})

else:
    st.info("👈 Upload an image to get started.")
