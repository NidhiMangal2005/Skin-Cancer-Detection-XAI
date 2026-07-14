import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import pandas as pd
from supabase import create_client


MODEL_PATH = "final_efficientnet_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)


st.set_page_config(
    page_title="Skin Cancer Detection XAI",
    page_icon="🩺",
    layout="wide"
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


@st.cache_resource
def init_supabase():
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    return create_client(supabase_url, supabase_key)


model, class_names = load_model_and_classes()
supabase = init_supabase()


st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #d1d5db;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #374151;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #3f3f16;
        padding: 16px;
        border-radius: 10px;
        color: #fff7c2;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_top_predictions(preds, top_n=3):
    probs = preds[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    top_predictions = []

    for idx in top_indices:
        top_predictions.append({
            "class": class_names[int(idx)],
            "confidence": round(float(probs[idx] * 100), 2)
        })

    return top_predictions


def predict_image(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)

    pred_index = int(np.argmax(preds[0]))
    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index] * 100)

    top_predictions = get_top_predictions(preds, top_n=3)

    return pred_class, confidence, top_predictions, img_array


def make_gradcam_heatmap(img_array):
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer_name = "top_conv"

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(img_array)

        pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)

        predictions = model.layers[-1](pooled_features)

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) != 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_gradcam(original_image, heatmap):
    original_image = original_image.convert("RGB").resize(IMG_SIZE)
    original_array = np.array(original_image)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap_color * 0.4 + original_array
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img


def save_prediction_to_db(user_email, image_name, pred_class, confidence, top_predictions):
    record = {
        "user_email": user_email,
        "image_name": image_name,
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 2),
        "top_predictions": top_predictions
    }

    response = supabase.table("predictions").insert(record).execute()
    return response


def fetch_user_history(user_email):
    response = (
        supabase
        .table("predictions")
        .select("*")
        .eq("user_email", user_email)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""


with st.sidebar:
    st.title("🩺 Skin Cancer XAI")

    if not st.session_state.logged_in:
        st.subheader("Login")
        email = st.text_input("Enter your email")

        if st.button("Login"):
            if email and "@" in email:
                st.session_state.logged_in = True
                st.session_state.user_email = email.strip().lower()
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Please enter a valid email address.")
    else:
        st.success(f"Logged in as {st.session_state.user_email}")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.rerun()

    st.divider()

    page = st.radio(
        "Navigation",
        ["Home", "New Scan", "History", "About"]
    )


if page == "Home":
    st.markdown('<div class="main-title">Explainable AI-Based Skin Cancer Detection</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="subtitle">A machine learning web application for skin lesion classification using EfficientNetB0 and Grad-CAM explainability.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">
        This application is a decision-support prototype. It does not replace professional medical diagnosis.
        Please consult a dermatologist for medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "EfficientNetB0")

    with col2:
        st.metric("Classes", "7 lesion types")

    with col3:
        st.metric("Explainability", "Grad-CAM")

    st.subheader("Main Features")

    st.write(
        """
        - Upload dermoscopic skin lesion images
        - Predict lesion category
        - Show confidence score
        - Display top 3 predictions
        - Generate Grad-CAM heatmap
        - Save prediction history for logged-in users
        """
    )


elif page == "New Scan":
    st.title("New Skin Lesion Scan")

    if not st.session_state.logged_in:
        st.warning("Please login from the sidebar to save prediction history.")

    uploaded_file = st.file_uploader(
        "Upload skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        pred_class, confidence, top_predictions, img_array = predict_image(image)

        with col2:
            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {pred_class}")
            st.metric("Confidence Score", f"{confidence:.2f}%")

            st.subheader("Top 3 Predictions")
            for item in top_predictions:
                st.write(f"**{item['class']}** — {item['confidence']}%")

        heatmap = make_gradcam_heatmap(img_array)
        gradcam_image = overlay_gradcam(image, heatmap)

        st.subheader("Grad-CAM Explanation")
        st.write("The heatmap highlights the image regions that influenced the model prediction.")

        col3, col4 = st.columns([1, 1])

        with col3:
            st.image(image, caption="Original Image", use_container_width=True)

        with col4:
            st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)

        if st.session_state.logged_in:
            if st.button("Save Result to History"):
                try:
                    save_prediction_to_db(
                        user_email=st.session_state.user_email,
                        image_name=uploaded_file.name,
                        pred_class=pred_class,
                        confidence=confidence,
                        top_predictions=top_predictions
                    )
                    st.success("Prediction saved to history successfully.")
                except Exception as e:
                    st.error("Could not save prediction to database.")
                    st.exception(e)


elif page == "History":
    st.title("Patient Prediction History")

    if not st.session_state.logged_in:
        st.warning("Please login from the sidebar to view your history.")
    else:
        try:
            history = fetch_user_history(st.session_state.user_email)

            if not history:
                st.info("No prediction history found yet.")
            else:
                df = pd.DataFrame(history)

                display_df = df[
                    [
                        "created_at",
                        "image_name",
                        "predicted_class",
                        "confidence"
                    ]
                ]

                display_df = display_df.rename(
                    columns={
                        "created_at": "Date",
                        "image_name": "Image Name",
                        "predicted_class": "Predicted Class",
                        "confidence": "Confidence (%)"
                    }
                )

                st.dataframe(display_df, use_container_width=True)

                st.subheader("Detailed Records")

                for record in history:
                    with st.expander(f"{record['predicted_class']} — {record['confidence']}%"):
                        st.write("**Image Name:**", record.get("image_name"))
                        st.write("**Prediction:**", record.get("predicted_class"))
                        st.write("**Confidence:**", str(record.get("confidence")) + "%")
                        st.write("**Created At:**", record.get("created_at"))
                        st.write("**Top Predictions:**")
                        st.json(record.get("top_predictions"))

        except Exception as e:
            st.error("Could not fetch history.")
            st.exception(e)


elif page == "About":
    st.title("About This Project")

    st.write(
        """
        This project is an Explainable AI-Based Skin Cancer Detection system.
        It uses a trained EfficientNetB0 deep learning model to classify dermoscopic skin lesion images
        into seven lesion categories.
        """
    )

    st.subheader("Technology Stack")

    st.write(
        """
        - Python
        - TensorFlow / Keras
        - EfficientNetB0
        - OpenCV
        - Grad-CAM
        - Streamlit
        - Supabase
        - GitHub
        - Streamlit Community Cloud
        """
    )

    st.subheader("Disclaimer")

    st.warning(
        "This system is built for educational and research purposes only. "
        "It should not be used as a substitute for professional medical diagnosis."
    )
