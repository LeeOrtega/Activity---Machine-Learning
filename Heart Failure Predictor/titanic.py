import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import pickle


st.title('PREDICT YOUR HEART FAILURE SURVIVAL')
img = Image.open('heartfailure.jpg')
st.image(img,width=600, channels='RGB',caption=None)



model = load_model('heart_failure_model.h5')
history = pickle.load(open('training_history','rb'))

def DEATH_EVENT():
    Anemia = st.sidebar.selectbox('anaemia',[1,0])
    Diabetes = st.sidebar.selectbox('diabetes',[1,0])
    ejection_Fraction = st.sidebar.slider('ejection_fraction', 10,100)
    High_Blood_Pressure = st.sidebar.selectbox('high_blood_pressure', [1,0])
    sex = st.sidebar.selectbox('sex', ('male','female'))
    age = st.sidebar.slider('age',0,80)
    creatinine_phosphokinase = st.sidebar.slider('creatinine_phosphokinase', 0,2600)
    platelets = st.sidebar.slider('platelets', 20000,400000)
    serum_creatinine = st.sidebar.slider('serum_creatinine', 0,5)
    serum_sodium = st.sidebar.slider('serum_sodium', 100,150)
    smoker = st.sidebar.selectbox('smoking', [0,1])
    time = st.sidebar.slider('time', 0,300)
    
    if sex == 'Male':
        sex = 1
    else:
        sex = 0

    data = [[Anemia, Diabetes, sex, ejection_Fraction, High_Blood_Pressure, age, creatinine_phosphokinase, platelets, serum_creatinine, serum_sodium, smoker, time]]
    data = tf.constant(data)
    return data

survive_or_not=DEATH_EVENT()
prediction = model.predict(survive_or_not)
pred = [round(x[0]) for x in prediction]
survive_or_not = np.array(survive_or_not).reshape(1, -1)

if pred ==[0]:
    st.write('You suffered and died')
else:
    st.write('You survived')


def plot_data():
    loss_train=history['loss']
    loss_val = history['val_loss']
    epochs = range(1,120)
    fig, ax = plt.subplots()
    ax.scatter([0.25],[0.25])
    plt.plot(epochs, loss_train,'g', label = 'Training Loss')
    plt.plot(epochs, loss_val,'b', label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
    loss_train =history['acc']
    loss_val = history['val_acc']
    epochs = range(1,120)
    fig, ax = plt.subplots()
    ax.scatter([1], [1])
    plt.plot(epochs, loss_train, 'g', label='Training Accuracy')
    plt.plot(epochs, loss_val, 'b', label='Validation Accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    
plot_data()