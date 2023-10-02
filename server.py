from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/analyze_ecg', methods=['POST'])
def analyze_ecg():
    # Get image from the request
    image = request.files['ecg_image']

    # Save the image temporarily
    image_path = "temp_image.png"
    image.save(image_path)

    # Call the basic_ecg_interpretation function
    result = basic_ecg_interpretation(image_path)

    # Return the result
    return jsonify({"result": result})

def basic_ecg_interpretation(image_path):
    # Load the ECG image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(thresholded, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
    
    if lines is None:
        return "Unable to detect lines in the ECG image."
    
    # Count the number of vertical lines (QRS complexes)
    vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]
    num_vertical_lines = len(vertical_lines)
    
    # Basic diagnosis based on the count of detected lines
    if num_vertical_lines < 5:
        return "Bradycardia (Slow heart rate) - This is a rudimentary diagnosis. Please consult a professional."
    elif num_vertical_lines > 12:
        return "Tachycardia (Fast heart rate) - This is a rudimentary diagnosis. Please consult a professional."
    else:
        return "Normal heart rate (approximation) - This is a rudimentary diagnosis. Please consult a professional."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0' ,port=3000)
