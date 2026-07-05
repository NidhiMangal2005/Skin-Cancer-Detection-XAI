
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = "final_efficientnet_model.keras"
CLASS_NAMES_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("Explainable AI-Based Skin Cancer Detection")
st.write(
    "Upload a dermoscopic skin lesion image. "
    "The system predicts the lesion category and displays a Grad-CAM heatmap."
)

st.warning(
    "This application is only a decision-support tool. "
    "It does not replace professional medical diagnosis."
)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index] * 100
    return pred_class, confidence, img_array

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

        # The final layer of our Sequential model is Dense(7)
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

uploaded_file = st.file_uploader(
    "Upload skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    pred_class, confidence, img_array = predict_image(image)

    st.subheader("Prediction Result")
    st.write("**Predicted Class:**", pred_class)
    st.write("**Confidence Score:**", f"{confidence:.2f}%")

    heatmap = make_gradcam_heatmap(img_array)
    gradcam_image = overlay_gradcam(image, heatmap)

    st.subheader("Grad-CAM Explanation")
    st.write(
        "The heatmap highlights the regions that influenced the model prediction."
    )
    st.image(gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
