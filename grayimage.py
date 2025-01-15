import cv2
import pandas as pd
import pytesseract
import re
import fitz  # PyMuPDF for handling PDF files
import os
from PIL import Image
import numpy as np

# Specify the path to Tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to preprocess the image for better OCR accuracy
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return thresh


# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image).lower()
    return text


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution (zoom factor)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Preprocess image to enhance text visibility
        img = img.convert('L')  # Convert to grayscale
        img = np.array(img)
        preprocessed_img = preprocess_image(img)

        # Extract text using Tesseract
        text += pytesseract.image_to_string(preprocessed_img)  # OCR to extract text
    return text


# Function to extract test results using regex
def extract_test_results(text):
    # A more robust regex pattern to capture variations
    pattern = r"(creatinine|urea|chloride|sodium|blood urea nitrogen \(bun\)|glomerular filtration rate \(gfr\)|urine albumin):?\s*(\d+\.?\d*)"
    matches = re.findall(pattern, text)
    data = [{"Test Name": match[0], "Result": match[1]} for match in matches]
    return pd.DataFrame(data)


# Main function
def main():
    # Ask user for the file path
    file_path = input("Enter the file path (image or PDF): ")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Check the file extension
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
        # Extract text from image
        print("Processing image...")
        text = extract_text_from_image(file_path)
    elif file_extension.lower() == '.pdf':
        # Extract text from PDF
        print("Processing PDF...")
        text = extract_text_from_pdf(file_path)
    else:
        print("Unsupported file type.")
        return

    # Display the extracted text
    print("Extracted Text: ")
    print(text)

    # Extract test results from the text
    df = extract_test_results(text)

    # Display the extracted data
    if not df.empty:
        print("Extracted Test Results:")
        print(df)
        # Save to CSV file
        df.to_csv("extracted_test_results.csv", index=False)
        print("Extracted test results saved to 'extracted_test_results.csv'.")
    else:
        print("No test results found.")


# Run the main function
if __name__ == "__main__":
    main()
