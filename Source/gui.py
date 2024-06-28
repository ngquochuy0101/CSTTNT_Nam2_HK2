import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('D:\DoAn_CSTTNT\Source\model-cifar10.h5')

#dictionary to label all traffic signs class.
classes = ['chim', 'mèo', 'hưu', 'chó', 'ếch', 'ngựa']
         
#initialise GUI
top=tk.Tk()
top.geometry('500x500')
top.title('Nhận diện động vật')
top.configure(background='#ffffff')

label=Label(top,background='#ffffff', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
# predict classes
    pred_probabilities = model.predict(image)[0]
    pred = pred_probabilities.argmax(axis=-1)
    sign = classes[pred-2]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Nhận dạng",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#c71b20', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#c71b20', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Nhận dạng Động vật",pady=10, font=('arial',20,'bold'))
heading.configure(background='#ffffff',foreground='#364156')

heading1 = Label(top, text="Môn Học: Cơ sở trí tuệ nhân tạo",pady=10, font=('arial',20,'bold'))
heading1.configure(background='#ffffff',foreground='#364156')

heading2 = Label(top, text="Danh sách thành viên nhóm\nNguyễn Quốc Huy\nNguyễn Vũ Hào\nĐoàn Hoàng Long\nNguyễn Huỳnh Tài\nPhạm Tấn Khương",pady=5, font=('arial',20,'bold'))
heading2.configure(background='#ffffff',foreground='#364156')

heading.pack()
heading1.pack()
heading2.pack()
top.mainloop()
