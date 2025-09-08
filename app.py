from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("digit_model.h5")

@app.route("/")
def home():
    """Render the homepage with upload + camera UI"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return digit prediction"""
    file = request.files['file']  # Get uploaded file
    img = image.load_img(file, target_size=(28, 28), color_mode="grayscale")  # Resize & grayscale
    img_array = image.img_to_array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize
    
    prediction = np.argmax(model.predict(img_array))  # Get predicted digit
    return jsonify({"digit": int(prediction)})

if __name__ == "__main__":
    # IMPORTANT: Use 0.0.0.0 + port 7860 for Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860, debug=False)
