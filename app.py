from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("digit_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = image.load_img(file, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = np.argmax(model.predict(img_array))
    return jsonify({"digit": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)   # IMPORTANT for Hugging Face
