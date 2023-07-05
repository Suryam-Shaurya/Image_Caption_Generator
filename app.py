import sys

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import time

print('started')


def load_dependencies():
    model_30 = load_model('h5/final_model_2030.h5')
    model_8 = load_model('h5/best_model.h5')
    file = open('pkl/Flickr30k_tokenizer.pkl', 'rb')
    tokenizer_30 = pickle.load(file)
    file.close()
    file = open('pkl/Flickr8k_tokenizer.pkl', 'rb')
    tokenizer_8 = pickle.load(file)
    file.close()
    model_img = load_model('h5/Flickr8k_VGG16.h5')
    return model_30, model_8, tokenizer_30, tokenizer_8, model_img


max_length_30 = 80
max_length_8 = 37


if ("model_30" not in st.session_state.keys()) and ("model_8" not in st.session_state.keys()) and ("tokenizer_30" not in st.session_state.keys()) and ("tokenizer_8" not in st.session_state.keys()) and ("model_img" not in st.session_state.keys()):
    col1, col2, col3 = st.columns(3)
    with col2:
        gif_runner = st.image('img_gif/processing.gif')
        yo = st.title('Loading...')
    st.session_state["model_30"], st.session_state["model_8"], st.session_state["tokenizer_30"], st.session_state["tokenizer_8"], st.session_state["model_img"] = load_dependencies()
    yo.empty()
    gif_runner.empty()

model_30, model_8, tokenizer_30, tokenizer_8, model_img = st.session_state["model_30"], st.session_state["model_8"], st.session_state["tokenizer_30"], st.session_state["tokenizer_8"], st.session_state["model_img"]


st.markdown("# :blue[What's in the image?]")



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose = 0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


def generate_caption_with_image_path(model, image_path, max_length, tokenizer):
    image = load_img(image_path, target_size = (224, 224))
    print("Image found...")
    image_arr = img_to_array(image)
    print("Processing Image...")
    image_arr = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
    image_arr = preprocess_input(image_arr)
    feature = model_img.predict(image_arr, verbose = 0)
    y_pred = predict_caption(model, feature, tokenizer, max_length)
    print("In this Image...")
    y_pred = y_pred.replace('startseq ', '')
    y_pred = y_pred.replace(' endseq', '')
    print(y_pred)
    return y_pred


st.markdown('## :orange[Upload your Image file here...]')
image_path = st.file_uploader("Upload your Image file here...", type=['png', 'jpeg', 'jpg'], label_visibility='collapsed')

if image_path is not None:
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image_path)
    generating_caption = st.markdown('## :orange[Hmmm... Wait...:face_with_monocle:]')
    generated_caption_30 = generate_caption_with_image_path(model_30, image_path, max_length_30, tokenizer_30).capitalize() + '.'
    generated_caption_8 = generate_caption_with_image_path(model_8, image_path, max_length_8, tokenizer_8).capitalize() + '.'
    time.sleep(1)
    generating_caption.empty()
    st.markdown('## :green[What I can see is...]')
    st.write(f'### {generated_caption_30}')
    st.write(':orange[or may be...]')
    st.write(f'### {generated_caption_8}')

print('done')











