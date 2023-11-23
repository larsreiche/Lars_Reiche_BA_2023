import cv2
from keras.models import load_model
import numpy as np
from PIL import Image

classes = [
    "aubergine",
    "birne",
    "ingver",
    "moehren",
    "radischen",
    "selerie",
    "tomate",
    "zitrone",
    "zuchini",
    "zwetschgen"
]

model = load_model('obst.h5')

input_shape = model.input_shape[1:3]  # Annahme: Die Eingabegröße ist (None, 640, 480, 3)

def predict_image(image):
    image = image.resize((input_shape[1], input_shape[0]))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return predicted_class, classes[predicted_class]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    predicted_class, class_name = predict_image(pil_image)
    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
