import os
import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Label, messagebox
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from PIL import Image, ImageTk

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 28, 28, 1))
    return image

def train_model():
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    train_labels = to_categorical(train_labels)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3,), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
    model.save('fashion_mnist_model.h5')
    return model

def classify_image(image_path, model):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction)

    classes = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    predicted_class = classes[predicted_label]

    return predicted_class

def load_image():
    global model
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class = classify_image(file_path, model)
        prediction_label.config(text=f'The input image is predicted to be a {predicted_class}.')

        user_feedback = messagebox.askyesno("Prediction Feedback", "Was the prediction correct?")
        if not user_feedback:
            print("Retraining the model with the updated data")
            model = train_model()

            predicted_class = classify_image(file_path, model)
            prediction_label.config(text=f'After retraining, the input image is predicted to be a {predicted_class}')
    else:
        print("No file selected.")

model_path = 'fashion_mnist_model.h5'
if not os.path.exists(model_path):
    print("Model not found. Training new model.")
    model = train_model()
else:
    print("Loading the existing trained model.")
    model = load_model(model_path)

root = Tk()
root.title("Image Classifier")

def open_image():
    load_image()

button_image = Image.open("C:\\Users\\mridu\\OneDrive\\ai and ml mtc\\images.png")
button_image = button_image.resize((500, 400))
button_photo = ImageTk.PhotoImage(button_image)

image_button = Button(root, image=button_photo, command=open_image)
image_button.pack()

prediction_label = Label(root, text="")
prediction_label.pack()

root.mainloop()

