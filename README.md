# Image Processing API - Fourier Transform Compression

This Flask-based API provides image processing using Fourier Transform compression techniques. 

## What This API Does

The API takes an uploaded image and:
1. Converts it to grayscale
2. Applies 2D Fourier Transform 
3. Compresses the image using different threshold values
4. Returns the original and compressed versions with compression statistics with thresholds at 0.1%, 1%, 5%, and 10% of maximum amplitude

## Setup Instructions

### 1. Install Dependencies

```bash
pip install requirements.txt
```

### 2. Environment Configuration

- `PORT`: Server port (default: 10000)

### 3. Run the Application

```bash
python app.py
```

## Usage Examples

Visit https://www.tringuyen.work/Blogs/2dae53ee-df66-4aed-9305-d1e7fbd0a0ea

### Supported File Formats

- JPEG/JPG
- PNG
- GIF
- BMP
- TIFF
- WebP

### Limitations

- Maximum file size: 10MB
- Images larger than 800px are automatically resized
- There will be a 30 seconds wait time to activate my Render server
