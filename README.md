# Describe.It

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ezizkerem-imagecaption-app-xzmysu.streamlitapp.com/)

An web-app generates natural language description for images utilizing neural networks. 

## Data

[Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) is used to train the model, which contains eight thousands of images and each image with 5 descriptions.

## Model

DenseNet 201 is used to extract the features from the image, then extracted features and captions are feed into neural network to train the model.

## Web Application

Click [here](https://ezizkerem-imagecaption-app-xzmysu.streamlitapp.com/) to interactive with the application.

## Repository Structure

1. [assets](./assets/)
    > Images used by the application.

2. [features](./features/)
    > Saved DenseNet model for extracting features from the image.

3. [image_cap](./image_cap/)
    > Saved neural network model for generating descriptions.

4. [app.py](app.py)
    > Code for the application.

5. [README.md](README.md)
    > The document you are currently viewing.

---

<div align="center">

<h4 id="tools">Tools & Libraries</h4>

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)


</div>