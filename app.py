import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load trained model
model = load_model("digit_model.h5")

# ---------- Preprocessing ----------
def preprocess_image(img):
    img = img.convert("L")  # grayscale
    img = np.array(img)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)

    return img

# ---------- Digit Segmentation + Prediction ----------
def segment_and_predict(img):
    processed = preprocess_image(img)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    annotated_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8:
            continue

        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(processed.shape[1], x + w + pad), min(processed.shape[0], y + h + pad)

        digit_crop = processed[y1:y2, x1:x2]
        digit_crop = cv2.resize(digit_crop, (28, 28), interpolation=cv2.INTER_AREA)
        digit_crop = digit_crop.astype("float32") / 255.0
        digit_crop = digit_crop.reshape(1, 28, 28)

        digit_regions.append((x1, y1, x2, y2, digit_crop))

    # Sort by rows (top-to-bottom) and then left-to-right within each row
    row_threshold = 50  # adjust if digits are spaced vertically
    digit_regions.sort(key=lambda r: (r[1] // row_threshold, r[0]))

    predictions = []
    for x1, y1, x2, y2, digit in digit_regions:
        pred = np.argmax(model.predict(digit, verbose=0))
        predictions.append((x1, y1, pred))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_img, str(pred), (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    annotated_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

    # Format output: group digits by rows, left-to-right
    if not predictions:
        output_text = "No digits detected"
    else:
        # Group by row
        rows = {}
        for x, y, pred in predictions:
            row_key = y // row_threshold
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x, pred))
        # Sort each row left-to-right
        output_text = ""
        for key in sorted(rows.keys()):
            row_digits = [str(pred) for x, pred in sorted(rows[key], key=lambda t: t[0])]
            output_text += "".join(row_digits) + "\n"
        output_text = output_text.strip()

    return output_text, annotated_img

# ---------- UI ----------
st.set_page_config(page_title="Digit Recognition", layout="wide")

# ---------- Custom Header ----------
st.markdown(
    """
    <div style="background-color:#ADD8E6; padding:10px; display:flex; align-items:center;">
        <a href="https://github.com/Sushant-Kumar-0028/Digit-Recognition-System" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" s
            width="35" style="margin-right:15px;">
        </a>
        <h2 style="margin:0; width:100%; text-align:center; color:black; font-weight:bold;">
            Handwritten Digit Recognition System
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(""); st.write(""); st.write("")

col1, col2 = st.columns(2)

# ---------- Upload section ----------
with col1:
    st.subheader("📂 Upload Image")
    uploaded_file = st.file_uploader("Upload PNG/JPEG", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Submit", key="upload_submit"):
            result, annotated = segment_and_predict(img)
            st.image(annotated, caption="Detected Digits with Bounding Boxes", use_container_width=True)

# ---------- Camera section ----------
with col2:
    st.subheader("📸 Use Camera")

    # Helper message for better accuracy
    st.info("✍️ Please write numbers on an unruled / plain white page for best results.")

    # Initialize camera state
    if 'camera_active' not in st.session_state:
        st.session_state['camera_active'] = False

    # Activate / Close Camera buttons
    if not st.session_state['camera_active']:
        if st.button("Activate Camera", key="activate_camera"):
            st.session_state['camera_active'] = True
    else:
        if st.button("Close Camera", key="close_camera"):
            st.session_state['camera_active'] = False

    # Show camera input only if active
    if st.session_state['camera_active']:
        camera_file = st.camera_input("Take a picture", key="camera_input")

        if camera_file:
            cam_img = Image.open(camera_file)
            st.image(cam_img, caption="Captured Image", use_container_width=True)

            if st.button("Submit", key="camera_submit"):
                result, annotated = segment_and_predict(cam_img)
                # st.success(f"Recognized Digits:\n{result}")
                st.image(annotated, caption="Detected Digits with Bounding Boxes", use_container_width=True)

# Footer
st.markdown(
    """
    <div style="background:#f0f0f0;padding:8px;text-align:center;
    position:fixed;bottom:0;left:0;right:0;">
        <a href="https://www.linkedin.com/in/sushant-kumar-9a7b9a1b6/" 
        target="_blank">@Linkedin-Sushant-Kumar</a>
    </div>
    """,
    unsafe_allow_html=True,
)
