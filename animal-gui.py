import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants and Configuration
IMG_SIZE = 128
CATEGORIES = ["Carnivore", "Herbivore"]
MODEL_PATH = "animal_detection_model.keras"  # Update the file extension

# Load and Preprocess Dataset
def load_images(dataset_path):
    data, labels = [], []
    for category in CATEGORIES:
        category_path = os.path.join(dataset_path, category)
        label = CATEGORIES.index(category)
        for animal in os.listdir(category_path):
            animal_path = os.path.join(category_path, animal)
            for img_file in os.listdir(animal_path):
                img_path = os.path.join(animal_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img_resized)
                    labels.append(label)
    return np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0, to_categorical(labels, 2)

# Load and split data
data, labels = load_images("animal_recognition/animals")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the focal loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        cross_entropy = K.binary_crossentropy(y_true, y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed

# Check if the model exists, and load or create it
if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss()})
else:
    print("Creating a new model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Class output
    ])

    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])
    class_weight = {0: 2.0, 1: 1.0}
    
    # Save model checkpoints
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

    # Train the model
    model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), class_weight=class_weight, callbacks=[checkpoint])

    # Save the final model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Evaluate the model with additional metrics
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
print(classification_report(y_test_labels, y_pred_labels, target_names=CATEGORIES))

# GUI for Animal Detection
class AnimalDetectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Animal Detection")
        self.model = model
        self.init_gui()

    def init_gui(self):
        tk.Button(self.master, text="Load Image", command=self.load_image).pack()
        tk.Button(self.master, text="Load Video", command=self.load_video).pack()
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            carnivores_count = self.detect_and_classify_animals(img)
            messagebox.showinfo("Detection", f"Carnivores detected: {carnivores_count}")
            self.display_image(img)

    def load_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                carnivores_count = self.detect_and_classify_animals(frame)
                cv2.putText(frame, f"Carnivores: {carnivores_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Video", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

    def detect_and_classify_animals(self, img):
        carnivores_count = 0
        window_size = 128
        step_size = 64
        for y in range(0, img.shape[0] - window_size, step_size):
            for x in range(0, img.shape[1] - window_size, step_size):
                window = img[y:y + window_size, x:x + window_size]
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    continue
                window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                window_resized = cv2.resize(window_gray, (IMG_SIZE, IMG_SIZE))
                window_array = img_to_array(window_resized).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                prediction = self.model.predict(window_array)
                category = np.argmax(prediction)
                if category == 0:  # If it's a carnivore
                    carnivores_count += 1
                    cv2.rectangle(img, (x, y), (x + window_size, y + window_size), (255, 0, 0), 2)
        return carnivores_count

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = cv2.imencode('.png', img_rgb)[1].tobytes()
        tk_img = tk.PhotoImage(data=img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas.image = tk_img  # Prevent garbage collection

# Run the GUI
root = tk.Tk()
app = AnimalDetectionGUI(root)
root.mainloop()
