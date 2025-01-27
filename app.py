from flask import Flask, request, jsonify
import os
import pandas as pd
import pytesseract
import cv2
import pdfplumber
import re

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Specify the path to Tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update as needed

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image).lower()
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    extracted_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text.lower())
    return "\n".join(extracted_text)

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string().lower()
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = df.to_string().lower()
    return text

# First function to extract test results using regex
def extract_test_results(text):
    pattern = r"([a-z\s\(\),-]*(?:creatinine|sodium|potassium|chloride|electrolytes|blood urea nitrogen|bun|glomerular filtration rate|gfr)[a-z\s\(\),-]*)\s+(\d+\.?\d*)"
    matches = re.findall(pattern, text, re.MULTILINE)
    data = [{"Test Name": match[0].strip(), "Result": match[1]} for match in matches]
    return pd.DataFrame(data)

# Second function to extract test results using a different regex pattern
def extract_test_results1(text):
    pattern = r"([a-z\s\(\),-]+(?:creatinine|sodium|potassium|chloride|electrolytes|blood urea nitrogen|bun|glomerular filtration rate|gfr)[a-z\s\(\),-]*)\s+(\d+\.?\d*)?"
    matches = re.findall(pattern, text, re.MULTILINE)
    data1 = [{"Test Name": match[0].strip(), "Result": match[1]} for match in matches]
    return pd.DataFrame(data1)

# Route to display the file upload form
@app.route('/')
def upload_form():
    return '''<!doctype html>
    <html>
        <head>
            <title>Upload File</title>
        </head>
        <body>
            <h1>Upload File to Extract Test Results</h1>
            <form action="/upload" method="POST" enctype="multipart/form-data">
                <label>Select a file:</label>
                <input type="file" name="file">
                <br><br>
                <button type="submit">Upload and Process</button>
            </form>
        </body>
    </html>
    '''

# Route to handle file upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Determine file type and process it
    _, file_extension = os.path.splitext(file.filename)
    text = None

    try:
        if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            text = extract_text_from_image(file_path)
        elif file_extension.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension.lower() == '.csv':
            text = extract_text_from_csv(file_path)
        elif file_extension.lower() in ['.xls', '.xlsx']:
            text = extract_text_from_excel(file_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Extract test results using the first function
        results_df = extract_test_results(text)

        # If no results found using the first function, try the second one
        if results_df.empty:
            results_df = extract_test_results1(text)

        if not results_df.empty:
            results = results_df.to_dict(orient='records')
            return jsonify({"message": "File processed successfully", "results": results})
        else:
            return jsonify({"message": "No test results found in the file"}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8000)

