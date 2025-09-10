# Handwritten Digit Recognition System

A Streamlit-based web application to recognize handwritten digits from images, using a Convolutional Neural Network (CNN). Users can upload images or take pictures via a camera and get predictions with bounding boxes.

Features

Recognizes handwritten digits from uploaded images or live camera input.
Displays detected digits with bounding boxes on the image.
Handles multi-line digits, displaying predictions in a top-to-bottom, left-to-right order per line.
Easy-to-use web interface built with Streamlit.

Project Structure
Digit-Recognition-System/
│
├── app.py                 # Main Streamlit app
├── digit_model.h5         # Trained CNN model for digit recognition
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

Dataset
MNIST Dataset
The model was trained on the MNIST handwritten digits dataset (28x28 grayscale images of digits 0–9).
Training dataset: 60,000 images
Testing dataset: 10,000 images

Model
Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input Shape: 28x28 grayscale images
Output: 10-class softmax (digits 0–9)
Accuracy: ~97% on MNIST test dataset

Known Issues / Limitations

Multi-line Digit Recognition: Accuracy decreases for images containing digits in multiple lines.
Variable Digit Spacing: Digits written too close together or too far apart may be merged or missed.
Handwriting Variability: Slanted, connected, or unusually sized digits may reduce accuracy.
Image Quality & Noise: Low contrast, shadows, or blur affect predictions.
Small Digits: Very small digits may be ignored due to thresholding in preprocessing.
Camera vs Uploaded Images: Differences in lighting, rotation, or skew may slightly affect prediction accuracy.

Potential Improvements:

Automatic multi-row detection with vertical clustering.
Advanced preprocessing (adaptive thresholding, morphology) to handle noisy images.
Skew/rotation correction for camera images.
Training on larger, diverse handwriting datasets for improved robustness.

Install dependencies

pip install -r requirements.txt

Run the Streamlit app
streamlit run app.pystreamlit run app.py --server.port 8080 --server.address 0.0.0.0

Access the app
Open the URL displayed in the terminal (usually http://localhost:8501) in your browser.

Usage

Upload Image: Click on “Upload PNG/JPEG” to select an image file containing handwritten digits.

Activate Camera: Click “Activate Camera” to take a live picture of handwritten digits.

Manual Crop: Select the area containing digits with the mouse before predicting.

Predict: Click the predict button to view recognized digits and annotated images.


Dependencies
Python >= 3.8
TensorFlow
OpenCV (opencv-python)
Pillow (PIL)
Streamlit (streamlit)
NumPy



Author

Sushant Kumar
LinkedIn- https://www.linkedin.com/in/sushant-kumar-9a7b9a1b6/
