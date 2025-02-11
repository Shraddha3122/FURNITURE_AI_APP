from flask import Flask, request, jsonify
import cv2
import numpy as np
from collections import Counter

app = Flask(__name__)

# Helper function to extract dominant color
def get_dominant_color(image, k=3):
    # Convert image to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.reshape((-1, 3))
    img_data = np.float32(img_data)

    # K-means clustering for color quantization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count dominant color
    counts = Counter(labels.flatten())
    dominant = palette[np.argmax(list(counts.values()))]

    # Return color as integer tuple
    return tuple(map(int, dominant))

# Helper function to calculate color contrast
def color_contrast(color1, color2):
    # Euclidean distance for color contrast
    return np.linalg.norm(np.array(color1) - np.array(color2))

# API endpoint for color analysis
@app.route('/analyze_colors', methods=['POST'])
def analyze_colors():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    # Load image using OpenCV
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Detect wall and furniture color (for simplicity, using entire image)
    wall_color = get_dominant_color(image)
    furniture_color = get_dominant_color(image[image.shape[0] // 2:, :])

    contrast = color_contrast(wall_color, furniture_color)

    # Define a simple contrast threshold for recommendation
    contrast_threshold = 100.0
    recommendation = "Change furniture color" if contrast < contrast_threshold else "Colors are well contrasted"

    response = {
        "wall_color": wall_color,
        "furniture_color": furniture_color,
        "contrast_score": contrast,
        "recommendation": recommendation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)