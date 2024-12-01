import cv2
import numpy as np
from keras.models import load_model

# Set the size of the input images
img_size = (224, 224)

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

def preprocess_image(image):
    """Resize and normalize the image for model prediction."""
    img = cv2.resize(image, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_image(image_path):
    """Predicts if the given image contains cancer."""
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read the image file. Please check the path.")
        return
    img = preprocess_image(image)
    prediction = model.predict(img)
    label = 'Cancer' if prediction[0][0] > 0.5 else 'Non_Cancer'
    print(f"Prediction: {label}")
    # Show image with prediction
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_webcam():
    """Starts webcam feed and predicts cancer or non-cancer in real time."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_image(frame)
        prediction = model.predict(img)
        label = 'Cancer' if prediction[0][0] > 0.5 else 'Non_Cancer'
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Choose mode
mode = input("Enter 'image' to upload an image file or 'webcam' to use the webcam: ").strip().lower()

if mode == 'image':
    image_path = input("Enter the path to the image file: ")
    predict_image(image_path)
elif mode == 'webcam':
    predict_webcam()
else:
    print("Invalid option. Please choose 'image' or 'webcam'.")
