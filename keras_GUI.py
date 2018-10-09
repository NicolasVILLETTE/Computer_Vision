import keras
import numpy as np
import PIL
from PIL import Image, ImageTk
import Tkinter
from Tkinter import *
from tkFileDialog import askopenfile

from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

#
# import matplotlib.pyplot as plt

# MODELS = ['VGG16','InceptionV3','ResNet50','MobileNet']
#
# def load_model(models):
#
#     item = modelList.curselection()
#     modelList.get(item)
#
#     if modelList.get(item) == "VGG16":
#         keras_model = vgg16.VGG16(weights='imagenet')
#
#     elif modelList.get(item) == "InceptionV3":
#         keras_model = inception_v3.InceptionV3(weights='imagenet')
#
#     elif modelList.get(item) == "ResNet50":
#         keras_model = resnet50.ResNet50(weights='imagenet')
#
#     elif modelList.get(item) == "MobileNet":
#         keras_model = mobilenet.MobileNet(weights='imagenet')
#
#     return keras_model


keras_model = vgg16.VGG16(weights='imagenet')

# def file_preprocess(file):
#     pil_file = load_img(file, target_size=(224, 224))
#     numpy_file = img_to_array(pil_file)
#     batch_file = np.expand_dims(numpy_file, axis=0)
#     return batch_file

def predict():

    file = askopenfile()
    pil_file = load_img(file, target_size=(224, 224))
    numpy_file = img_to_array(pil_file)
    batch_file = np.expand_dims(numpy_file, axis=0)

    processed_file = vgg16.preprocess_input(batch_file)
    predictions = keras_model.predict(processed_file)
    labels = decode_predictions(predictions)
    labels_results.config(text=results_parsing(labels))

def results_parsing(list):
    object = list[0][0][1]
    probability = (list[0][0][2])*100
    return('{}: {:.2f}') .format(object, probability)


frame = Tk()
frame.title('VGG16 Object Detector')

canvas = Canvas(frame, height=224, width=224)
labels_results = Label(frame)
labels_results.pack(side='bottom')

# modelList = Listbox(frame, selectmode=SINGLE)
# modelList.bind('<<ListboxSelect>>', load_model)
# for models in MODELS:
#     modelList.insert(END, models)
# modelList.pack(side = "top")

def image_browser():
    filepath = askopenfile()
    return filepath

# def image_display():
#     filepath = askopenfile()
#     imageTK = ImageTk.PhotoImage(PIL.Image.open(filepath))
#     canvas.create_image(100, 80, image=imageTK, anchor="nw")
#     canvas.imageTK = imageTK
#     canvas.pack(side = 'right', expand=True, fill=BOTH)

# fileButton = Button(frame, text='Browse', command=image_display)
# fileButton.pack(side = 'bottom')

processButton = Button(frame, text='Process File', command=predict)
processButton.pack(side = 'bottom')


frame.mainloop()
