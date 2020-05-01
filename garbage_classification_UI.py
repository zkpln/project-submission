from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# import tkinter
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
global img_name
global instruction
global pic
global root

def predict_img(img_path):
    # img_path = 'test/test1.jpeg'

    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)

    img=np.array(img)/255.0

    model = load_model('/Users/weiyang/CSE574/model7.h5')

    p=model.predict(img[np.newaxis, ...])

    labels = {0:'Cardboard',1:'Glass',2:'Metal',3:'Paper',4:'Plastic',5:'Trash'}

    predicted_class = labels[np.argmax(p[0], axis=-1)]

    return predicted_class

def openImg():
    img_path = filedialog.askopenfilename(title='Select Image')

    result = predict_img(img_path)

    new_img = ImageTk.PhotoImage(Image.open(img_path).resize((300,300)))
    img_show.config(image = new_img)
    img_show.photo_ref = new_img

    predict_result.config(text="The Classification Result Is : " + result)

if __name__ == '__main__':
    # img_path = 'logo.jpeg'
    img_path = '/Users/weiyang/CSE574/logo.jpeg'
    root = Tk()
    root.title('Garbage Classification')
    root.geometry('700x500')

    # img = PhotoImage(Image.open(img_path))
    img = ImageTk.PhotoImage(Image.open(img_path))

    instruction = Label(root, text="Please Open A Image To Make Classification",font=('Times',32))
    instruction.pack()

    img_show = Label(root,image = img)
    img_show.pack(expand = "yes")

    predict_result = Label(root,text="",font=('Times',25))
    predict_result.pack()

    open_file = Button(root,text="Open",command = openImg)
    open_file.pack()

    root.mainloop()
