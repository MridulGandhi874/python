import os
import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Label, Canvas, messagebox, ttk, Scale
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image, ImageTk
import threading

class CustomProgressBar(ttk.Frame):
    def __init__(self, master, maximum):
        super().__init__(master)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate", maximum=maximum)
        self.progress.pack(pady=10)
        self.pack()

    def update_progress(self, value):
        self.progress['value'] = value
        self.update()

class FashionImageClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Fashion Image Classifier")
        self.root.attributes('-fullscreen', True)
        self.root.geometry("1000x600")

        self.canvas = Canvas(root, width=600, height=600, bg='white')
        self.canvas.pack(side="left", fill="both", expand=True)

        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(side="right", fill="y")

        self.open_button = Button(self.button_frame, text="Choose Video Source", command=self.load_video_and_process)
        self.open_button.pack(pady=10)

        self.clear_button = Button(self.button_frame, text="Clear Display", command=self.clear_display)
        self.clear_button.pack(pady=10)

        self.exit_button = Button(self.button_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)

        self.predicted_label_text = Label(self.button_frame, text="", font=('Helvetica', 14))
        self.predicted_label_text.pack(pady=10)

        self.class_probabilities_text = Label(self.button_frame, text="", font=('Helvetica', 12))
        self.class_probabilities_text.pack(pady=10)

        self.predicted_class = None
        self.model = None
        self.stop_video = False
        self.pause_video = False
        self.stop_button = None
        self.pause_button = None
        self.progress_bar = None
        self.video_cap = None
        self.total_frames = None
        self.current_frame = 0

    def preprocess_image(self, image, resize=True, normalize=True, grayscale=True):
        if grayscale and len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if resize:
            image = cv2.resize(image, (28, 28))
        if normalize:
            image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def train_model(self):
        (train_images, train_labels), (_, _) = fashion_mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
        train_labels = to_categorical(train_labels)

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3,), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
        self.model.save('fashion_mnist_model.h5')

        messagebox.showinfo("Model Evaluation", f"Model Training Completed.\nAccuracy: {history.history['accuracy'][-1]}, Loss: {history.history['loss'][-1]}")

    def classify_image(self, image, current_frame):
        if self.model is None:
            messagebox.showerror("Error", "Model is not loaded.")
            return
        input_image = self.preprocess_image(image)
        prediction = self.model.predict(input_image)
        predicted_label = np.argmax(prediction)

        classes = {
            0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
            5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
        }
        self.predicted_class = classes[predicted_label]

        prediction_text = f"The input image is predicted to be a {self.predicted_class}."
        self.predicted_label_text.config(text=prediction_text)

        class_probabilities = "\n".join([f"{classes[i]}: {prob}" for i, prob in enumerate(prediction[0])])
        self.class_probabilities_text.config(text="Class Probabilities:\n" + class_probabilities)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(700, 400, anchor='center', image=photo)
        self.canvas.image = photo

        self.progress_bar.update_progress(current_frame)

    def load_video_and_process(self):
        self.stop_video = False
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            if self.model is None:
                if not os.path.exists('fashion_mnist_model.h5'):
                    messagebox.showinfo("Training", "Training the model. This may take a while.")
                    self.train_model()
                try:
                    self.model = load_model('fashion_mnist_model.h5')
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load the model: {str(e)}")
                    return

            if not file_path.lower().endswith('.mp4'):
                messagebox.showerror("Error", "Invalid file type. Please select a .mp4 file.")
                return

            self.video_cap = cv2.VideoCapture(file_path)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar = CustomProgressBar(self.button_frame, self.total_frames)
            self.progress_bar.pack(pady=10)

            self.stop_button = Button(self.button_frame, text="Stop", command=self.stop_video_processing)
            self.stop_button.pack(pady=10)

            self.pause_button = Button(self.button_frame, text="Pause", command=self.pause_video_processing)
            self.pause_button.pack(pady=10)

            next_frame_button = Button(self.button_frame, text="Next Frame", command=self.next_frame)
            next_frame_button.pack(pady=10)

            prev_frame_button = Button(self.button_frame, text="Previous Frame", command=self.prev_frame)
            prev_frame_button.pack(pady=10)

            def process_frame():
                ret, frame = self.video_cap.read()
                if ret and not self.stop_video:
                    if not self.pause_video:
                        self.classify_image(frame, self.current_frame)
                        self.root.after(1, process_frame)
                    else:
                        self.root.after(100, process_frame)
                else:
                    self.video_cap.release()
                    cv2.destroyAllWindows()
                    self.stop_video = False
                    self.stop_button.destroy()
                    self.pause_button.destroy()
                    self.progress_bar.destroy()

            process_frame()
        else:
            messagebox.showerror("Error", "No file selected.")

    def pause_video_processing(self):
        self.pause_video = not self.pause_video

    def stop_video_processing(self):
        self.stop_video = True

    def clear_display(self):
        self.canvas.delete("all")
        self.predicted_label_text.config(text="")
        self.class_probabilities_text.config(text="")

    def exit_app(self):
        if messagebox.askokcancel("Exit", "Do you want to exit?"):
            self.root.destroy()

    def next_frame(self):
        if self.video_cap.isOpened():
            self.current_frame += 1
            if self.current_frame < self.total_frames:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.video_cap.read()
                if ret:
                    self.classify_image(frame, self.current_frame)

    def prev_frame(self):
        if self.video_cap.isOpened():
            self.current_frame -= 1
            if self.current_frame >= 0:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.video_cap.read()
                if ret:
                    self.classify_image(frame, self.current_frame)

root = Tk()
app = FashionImageClassifier(root)
root.mainloop()
