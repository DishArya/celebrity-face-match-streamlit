import types
import keras.engine.input_layer
import keras.utils

# Patch before any keras_vggface usage
import sys
import types

# Create a dummy module keras.engine.topology
import keras.engine.input_layer
import keras.utils.layer_utils

topology = types.ModuleType("keras.engine.topology")
topology.InputLayer = keras.engine.input_layer.InputLayer
topology.get_source_inputs = keras.utils.layer_utils.get_source_inputs

sys.modules['keras.engine.topology'] = topology

# Now import keras_vggface
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


detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embeded.pkl','rb'))
filenames = pickle.load(open('names.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    if not results:
        st.error("No face detected in the image.")
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which Celebrity You look Alike?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        if features is not None:
            index_pos = recommend(feature_list, features)
            matched_path = Path(filenames[index_pos]).as_posix()
            predicted_actor = " ".join(Path(matched_path).stem.split('_'))

            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Uploaded Image')
                st.image(display_image)

            with col2:
                st.header(f"Seems like {predicted_actor}")
                if Path(matched_path).exists():
                    st.image(matched_path, width=300)
                else:
                    st.error(f"Matched image not found: {matched_path}")

