from ultralytics import YOLO
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the model once when the server starts
model = YOLO("best.pt")

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    try:
        with open("index.html") as file:
            return file.read()
    except Exception as e:
        return str(e), 500

@app.route("/detect", methods=["POST"])
def detect():
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    passes it through YOLOv8 object detection
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding
    boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    try:
        buf = request.files["image_file"]
        boxes = detect_objects_on_image(Image.open(buf.stream))
        return Response(
          json.dumps(boxes),
          mimetype='application/json'
        )
    except Exception as e:
        return str(e), 500

def detect_objects_on_image(image):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: Input image file stream
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

def convert_to_dataframe(detection_results):
    data = []
    for result in detection_results:
        x1, y1, x2, y2, label, probability = result
        data.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label, 'probability': probability})
    df = pd.DataFrame(data)
    return df

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        detection_results = request.json['detection_results']
        df = convert_to_dataframe(detection_results)
        file_name = 'uploaded_image_analysis'

        # Add analysis and visualization code here
        bio_degradable = ['plant', 'animal_fish', 'animal_starfish', 'animal_shells',
                          'animal_crab', 'animal_eel', 'animal_etc', 'trash_fabric',
                          'trash_paper', 'trash_rubber', 'trash_wood']
        df['bio_degradable'] = df['label'].apply(lambda x: 1 if x in bio_degradable else 0)

        # Visualization
        counts = df['bio_degradable'].value_counts()
        labels = ['Non-BioDegradable' if i == 0 else 'BioDegradable' for i in counts.index]

        plt.figure(figsize=(20, 10))
        plt.title(f'{file_name}', size=20)
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.suptitle('Biodegradable vs Non-Biodegradable Waste')
        plt.legend()
        plt.savefig(f'{file_name}.png')
        plt.close()  # Close the plot to free up memory

        return jsonify({'message': 'Analysis complete', 'file_name': file_name})
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
