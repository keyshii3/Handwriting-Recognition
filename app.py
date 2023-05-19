from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import io
import cv2

app = Flask(__name__)

# Load the model
from keras.models import load_model
model = load_model('model12.h5', compile=False)

# Define the class labels
classes = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']

# Define the function to preprocess the image
def preprocess_image(image):
    # Get the input shape of the model
    input_shape = model.layers[0].input_shape[1:3]
    # Resize the image to the input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Invert the pixel values
    image = 255 - image
    # Normalize the pixel values
    image = image / 255.0
    # Reshape the image to a 4D array
    image = image.reshape(1, 224, 224, 1)
    return image

# Define the function to make a prediction
def predict_class(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Make a prediction
    prediction = model.predict(image)
    # Get the class label with the highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = classes[predicted_class_index]
    return predicted_class_name, prediction[0]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Read the file as an image
        image = Image.open(io.BytesIO(file.read()))
        # Convert the image to grayscale
        image = image.convert('L')
        # Make a prediction
        predicted_class_name, probabilities = predict_class(image)
        # Return the result
        return render_template('index.html', predicted_class_name=predicted_class_name, probabilities=probabilities)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)