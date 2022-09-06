import pickle
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# load tensorflow model
model = load_model('image_cap')

#load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle) 

uploaded_file = st.file_uploader("Choose an image...")

img = load_img(uploaded_file, color_mode='rgb', target_size=(224, 224))

#instantiate variables
feature = np.array([])

def display(image_file):
    img = load_img(image_file, color_mode='rgb', target_size=(224, 224))
    img = img_to_array(img)
    img = img/255 # change from rgb values 0-255 to 0-1
    fig = plt.figure() 
    plt.imshow(img)
    st.pyplot(fig)
    return img

def image_features(image_file):
    global feature
    model = DenseNet201()
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    image_file = np.expand_dims(image_file, axis=0) ##
    feature = fe.predict(image_file, verbose=0)
    st.write(feature)

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def generate_caption():
    # global caption, feature
    caption = ''
    for _ in range(20):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], 200)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)
                
        word = idx_to_word(y_pred, tokenizer)
        caption+= " " + word
    return caption
    

if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    image = Image.open(uploaded_file)	
    st.image(uploaded_file, caption='Input Image', use_column_width=True)	
    img = display(uploaded_file)
    image_features(img)
    st.write(img.shape)
    generate = st.button('generate')

if generate:
    st.write(generate_caption())
    