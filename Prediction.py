from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLO model once
model_path = r"C:\Users\Nishu Baviskar\OneDrive\Desktop\Try Bone Fracture\Yolov8 Models\m_model_epochs_100_imgsize_640_batchsize_32\runs\detect\yolov8n_custom\weights\best.pt"
model = YOLO(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ''
    output_path = ''

    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            file.save(img_path)

            img = cv2.imread(img_path)
            img_predict = img.copy()
            H, W, _ = img.shape

            results = model(img_predict)[0]
            fracture_detected = False

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > 0.5:
                    fracture_detected = True
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    class_name = results.names[int(class_id)].upper()
                    cv2.rectangle(img_predict, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_predict, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            prediction_text = "FRACTURED ✅" if fracture_detected else "NORMAL 🦴"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
            cv2.imwrite(output_path, img_predict)

    return render_template('index.html', prediction=prediction_text, output_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)






