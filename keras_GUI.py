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

keras_model = vgg16.VGG16(weights='imagenet')

class MainApplication:
    def __init__(self, master):
        self.master = master
        master.title("VGG16 Object Detector")

        self.label = Label(master)
        self.label.pack(side='bottom', fill='x')

        self.process_button = Button(master, text='Process File', command=self.predict)
        self.process_button.pack(side='bottom')

    def predict(self):
        file = askopenfile()
        pil_file = load_img(file, target_size=(224, 224))
        numpy_file = img_to_array(pil_file)
        batch_file = np.expand_dims(numpy_file, axis=0)

        processed_file = vgg16.preprocess_input(batch_file)
        predictions = keras_model.predict(processed_file)
        labels = decode_predictions(predictions)
        self.label.config(text=self.results_parsing(labels))

    def results_parsing(self, list):
        object = list[0][0][1]
        probability = (list[0][0][2])*100
        return('{}: {:.2f}') .format(object, probability)

def main():

    root = Tk()
    main_app = MainApplication(root)
    root.mainloop()

if __name__ == '__main__':
    main()
