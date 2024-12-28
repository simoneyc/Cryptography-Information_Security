# Cryptography-Information_Security

Here are my Homeworks & Project


## Project: Image-based Message Encoding and Decoding 
<img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExaDF6cGR5NTc2Y3A5enczMmVuMmR2N2d1czluamVkOW56NTh2MDBibCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/EoB3yOTwSck4gw1qQj/giphy.gif" width="120"/>

## Overview
This project provides a system for embedding and extracting secret messages in images using customized image processing and encoding techniques. It ensures minimal visual distortion in the original image while maintaining the fidelity of the embedded message.

## Features
- **Message Encoding**: Embeds a secret message into an image with minimal distortion.
- **Message Decoding**: Extracts and reconstructs the embedded message from an encoded image.
- **Image Comparison**: Validates image fidelity by comparing the original and encoded images.

## File Structure
- **`encode.py`**: Encodes a secret message into an input image.
  - Reads an input image.
  - Calculates prediction errors and selects regions for embedding.
  - Embeds the message and necessary parameters into the image.
  - Outputs an encoded image (`encoded_image.png`).

- **`decode.py`**: Decodes the secret message from an encoded image.
  - Reads the encoded image.
  - Extracts the embedded parameters.
  - Reconstructs the message based on extracted data.

- **`compare.py`**: Compares the original and encoded images.
  - Generates and displays histograms of grayscale values for both images.
  - Validates that grayscale invariance is maintained.

- **`functions.py`**: Provides utility functions used across the scripts, including:
  - Grayscale conversion (`cvtGray`).
  - Prediction error calculation (`PEs`).
  - Pixel value prediction (`predictV`).
  - Peak Signal-to-Noise Ratio calculation (`PSNR`).
  - Embedding message logic (`embedMsg`).

## Requirements
- Python 3.8+
- Required libraries:
  - OpenCV
  - NumPy
  - Matplotlib

Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Usage
### Encoding
1. Place your input image (`input.jpg`) in the project directory.
2. Edit the message to be encoded in `encode.py` (variable `msg`).
3. Run the encoder:
   ```bash
   python encode.py
   ```
4. The encoded image will be saved as `encoded_image.png`.

### Decoding
1. Ensure the encoded image (`encoded_image.png`) is in the project directory.
2. Run the decoder:
   ```bash
   python decode.py
   ```
3. The decoded message will be displayed in the terminal.

### Comparison
1. Place the original image (`input.jpg`) and the encoded image (`encoded_image.png`) in the project directory.
2. Run the comparator:
   ```bash
   python compare.py
   ```
3. Grayscale histograms and comparison results will be displayed and saved as `Grayscale.jpg`.

## Notes
- The `functions.py` file is required for auxiliary functions used across the scripts.
- Ensure images are in RGB format and appropriately sized for embedding.
- The size of the image must accommodate the length of the message to be embedded.

## Example
Encoding the message `welovecryptography` into an image:
1. Original image:
   ![Original Image](input.jpg)
2. Encoded image:
   ![Encoded Image](encoded_image.png)
3. Grayscale histogram comparison:
   ![Grayscale Histogram](Grayscale.jpg)

## Author
This project was developed to demonstrate an innovative approach to image-based steganography using Python. Contributions and suggestions are welcome!
