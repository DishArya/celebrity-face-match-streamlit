# 👇 Patch for keras_vggface compatibility with Keras 2.10
import sys
import types
import keras.engine.input_layer
import keras.utils.layer_utils

# Create and inject dummy keras.engine.topology module
topology = types.ModuleType("keras.engine.topology")
topology.InputLayer = keras.engine.input_layer.InputLayer
topology.get_source_inputs = keras.utils.layer_utils.get_source_inputs
sys.modules['keras.engine.topology'] = topology

# ✅ Now import keras_vggface
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from pathlib import Path

# Load face detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Load precomputed embeddings and filenames
feature_list = pickle.load(open('embeded.pkl', 'rb'))
filenames = pickle.load(open('names.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        uploads_dir = Path('uploads')
        uploads_dir.mkdir(exist_ok=True)
        file_path = uploads_dir / uploaded_image.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

def extract_features(img_path, model, detector):
    img = cv2.imread(str(img_path))
    results = detector.detect_faces(img)
    if not results:
        st.error("No face detected in the image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face).resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which Celebrity You Look Alike?')

uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    saved_path = save_uploaded_image(uploaded_image)
    if saved_path:
        display_image = Image.open(uploaded_image)
        features = extract_features(saved_path, model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            matched_path = Path(filenames[index_pos])
            predicted_actor = " ".join(matched_path.stem.split('_'))  # e.g., Tom_Cruise → Tom Cruise

            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Uploaded Image')
                st.image(display_image)

            with col2:
                st.header(f"Seems like {predicted_actor}")
                if matched_path.exists():
                    st.image(str(matched_path), width=300)
                else:
                    st.error(f"Matched image not found: {matched_path}")
