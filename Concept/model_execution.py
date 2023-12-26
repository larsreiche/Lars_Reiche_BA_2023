import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

# Anwendungsparameter-------------------------------------
model = load_model('og_model.h5')
classes = [
    "klasse-1",
    "klasse-2",
    # ....
    "klasse-n"
]
cap = cv2.VideoCapture(0)
# Alternative für DroidCam
# cap = cv2.VideoCapture('http://178.178.178.20:4747/video')


# Klassifikation -------------------------------------------
input_shape = model.input_shape[1:3]

def predict_image(image):
    image = image.resize((input_shape[1], input_shape[0]))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return predicted_class, classes[predicted_class]

while True:
    ret, frame = cap.read()
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    predicted_class, class_name = predict_image(pil_image)
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Webcam', frame)

cap.release()
cv2.destroyAllWindows()
