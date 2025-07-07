# app.py

import base64
import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# 1. Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# 2. Core Image Processing Functions
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
        # Open and convert image
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # Resize if too large to prevent memory issues
        max_size = 800
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        img_np = np.array(img)
        
        # Convert to grayscale
        gray_img = rgb2gray(img_np)
        
        print(f"Processing image of size: {gray_img.shape}")

        # Generate and encode the original grayscale image
        plt.figure(figsize=(6, 6))
        plt.imshow(gray_img, cmap='gray')
        plt.axis('off')
        plt.title("Original Grayscale")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        original_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Perform Fourier Transform
        F = np.fft.fft2(gray_img)
        F_shifted = np.fft.fftshift(F)

        # Process for different thresholds
        threshold_percentages = [0.001, 0.01, 0.05, 0.1]
        thresholds = [np.max(np.abs(F)) * p for p in threshold_percentages]
        
        results = []

        for i, threshold in enumerate(thresholds):
            try:
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
                plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
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
                
                print(f"Processed threshold {threshold_percentages[i]*100:.1f}%")
                
            except Exception as e:
                print(f"Error processing threshold {threshold_percentages[i]*100:.1f}%: {e}")
                continue

        return {
            "original_grayscale_base64": f"data:image/png;base64,{original_base64}",
            "processed_results": results
        }
        
    except Exception as e:
        print(f"Error in process_image_and_get_results: {e}")
        return {"error": f"Could not process image: {str(e)}"}

# 3. Define API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "message": "Image processing API is running",
        "version": "1.0.0"
    })

@app.route('/api/process', methods=['POST', 'OPTIONS'])
def handle_image_upload():
    """Endpoint to upload an image and get processed results."""
    
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check file size (limit to 10MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return jsonify({"error": "File too large. Maximum size is 10MB."}), 400

        # Process the image
        print(f"Processing file: {file.filename}, size: {file_size} bytes")
        
        img_bytes = file.read()
        output_data = process_image_and_get_results(img_bytes)
        
        if "error" in output_data:
            return jsonify(output_data), 500
            
        print("Image processing completed successfully")
        return jsonify(output_data)
        
    except Exception as e:
        print(f"Error in handle_image_upload: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# 4. Run the App
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
