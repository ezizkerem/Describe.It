import pickle
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np


st.title('Describe.It')

# header
col1, col2 = st.columns(2)

robot = Image.open('assets/Robot face-pana.png')
col1.image(robot, use_column_width=True)

col2.header("How does it work?")
col2.write('''Describle.It is designed to provide a 15-word description for an image uploaded by the user. To test it, you 
will need to: \n 1. Drag and drop an image in the box below \n 2. Wait for the AI to generate the information \n Easy right? Let's try it out...''')

uploaded_file = st.file_uploader(' ')

#instantiate variables
feature = np.array([])

def display(image_file):
    img = load_img(image_file, color_mode='rgb', target_size=(224, 224))
    img = img_to_array(img)
    img = img/255 # change from rgb values 0-255 to 0-1
    return img

def image_features(image_file):
    global feature
    # model = DenseNet201()
    # fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    fe = load_model('features')
    image_file = np.expand_dims(image_file, axis=0)
    feature = fe.predict(image_file, verbose=0)

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def generate_caption():

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) 

    model = load_model('image_cap')
    
    caption = ''
    for _ in range(15):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], 200)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)
                
        word = idx_to_word(y_pred, tokenizer)
        caption+= " " + word
    return caption
    

if uploaded_file is not None:
    st.image(uploaded_file, use_column_width=True)	
    img = display(uploaded_file)
    image_features(img)
    st.markdown(f'> {generate_caption()}')