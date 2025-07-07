# app.py

import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# 1. Initialize Flask App
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# 2. Core Image Processing Functions (adapted from your script)
def rgb2gray(rgb):
    """Convert an RGB image to grayscale."""
    return np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])

def apply_threshold(spectrum, threshold):
    """Set coefficients with amplitude below a threshold to zero."""
    modified_spectrum = spectrum.copy()
    modified_spectrum[np.abs(spectrum) < threshold] = 0
    return modified_spectrum

def process_image_and_get_results(img_bytes):
    """
    Takes image bytes, performs Fourier compression, and returns results.
    """
    try:
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        return {"error": f"Could not open image file: {e}"}

    gray_img = rgb2gray(img_np)

    # --- Generate and encode the original grayscale image ---
    plt.figure(figsize=(6, 6))
    plt.imshow(gray_img, cmap='gray')
    plt.axis('off')
    plt.title("Original Grayscale")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    original_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- Perform Fourier Transform ---
    F = np.fft.fft2(gray_img)
    F_shifted = np.fft.fftshift(F)

    # --- Process for different thresholds ---
    threshold_percentages = [0.001, 0.01, 0.05, 0.1]
    thresholds = [np.max(np.abs(F)) * p for p in threshold_percentages]
    
    results = []

    for i, threshold in enumerate(thresholds):
        # Apply threshold and get compressed image
        F_modified = apply_threshold(F, threshold)
        compressed_img = np.real(np.fft.ifft2(F_modified))

        # Calculate compression statistics
        non_zero_count = np.sum(np.abs(F_modified) > 0)
        total_count = F.size
        compression_ratio = non_zero_count / total_count if total_count > 0 else 0
        space_saved = 1 - compression_ratio

        # Plot the compressed image and save to buffer
        plt.figure(figsize=(6, 6))
        plt.imshow(compressed_img, cmap='gray')
        plt.title(f"Threshold: {threshold_percentages[i]*100:.1f}% of max\n"
                  f"Space Saved: {space_saved:.1%}")
        plt.axis('off')
        
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Encode image to Base64
        img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        
        # Store results
        results.append({
            "threshold": f"{threshold_percentages[i]*100:.1f}%",
            "image_base64": f"data:image/png;base64,{img_base64}",
            "compression_ratio": f"{compression_ratio:.3f}",
            "space_saved": f"{space_saved:.1%}"
        })

    return {
        "original_grayscale_base64": f"data:image/png;base64,{original_base64}",
        "processed_results": results
    }


# 3. Define the API Endpoint
@app.route('/api/process', methods=['POST'])
def handle_image_upload():
    """Endpoint to upload an image and get processed results."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        img_bytes = file.read()
        output_data = process_image_and_get_results(img_bytes)
        
        if "error" in output_data:
            return jsonify(output_data), 500
            
        return jsonify(output_data)

# 4. Run the App
if __name__ == '__main__':
    # Port 10000 is often recommended by Render for web services
    app.run(host='0.0.0.0', port=10000, debug=True)
