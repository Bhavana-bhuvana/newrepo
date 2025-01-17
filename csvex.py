import cv2
import pandas as pd
import pytesseract
import re
import os
from PIL import Image
import pdfplumber

# Specify the path to Tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image).lower()
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    extracted_text = []
    # Use pdfplumber to handle text-based PDFs
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text if it's a text-based page
            text = page.extract_text()
            text=text.lower()
            if text:
                extracted_text.append(text)
            else:
               print("no text found")

    return "\n".join(extracted_text)

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text


def clean_extracted_text(text):
    # Remove unwanted newlines and excess whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\n+', ' ', text)  # Remove line breaks (newlines)
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# Function to extract test results using regex
def extract_test_results(text):
    #pattern = r"(creatinine|urea|chloride|sodium|blood urea nitrogen \(bun\)|glomerular filtration rate \(gfr\)|Urine Albumin):?\s*(\d+\.?\d*)"
    #pattern = r"([a-z\s\(\),-]+(?:creatinine|sodium|potassium|cloride|electrolytes|blood urea nitrogen|bun|glomerular filtration rate|gfr)[a-z\s\(\),-]*)\s+(\d+\.?\d*)?"
    pattern = r"([a-z\s\(\),-]*(?:creatinine|sodium|potassium|chloride|electrolytes|blood urea nitrogen|bun|glomerular filtration rate|gfr)[\s\(\),-]*)\s+(\d+\.?\d*)"

    matches = re.findall(pattern, text, re.IGNORECASE)
    data = [{"Test Name": match[0].strip(), "Result": match[1]} for match in
            matches]
    return pd.DataFrame(data)


# Main function
def main():
    # Ask user for the file path
    file_path = input("Enter the file path (image, PDF, CSV, or Excel): ")

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
    elif file_extension.lower() == '.csv':
        # Extract text from CSV
        print("Processing CSV...")
        text = extract_text_from_csv(file_path)
    elif file_extension.lower() in ['.xls', '.xlsx']:
        # Extract text from Excel
        print("Processing Excel...")
        text = extract_text_from_excel(file_path)
    else:
        print("Unsupported file type.")
        return

    # Display the extracted text
    print("Extracted Text: ")
    print(text)

    # Extract test results from the text
    text=clean_extracted_text(text)
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
'''import cv2
import pandas as pd
import pytesseract
import re
import fitz  # PyMuPDF for handling PDF files
import os
from PIL import Image
from transformers import pipeline  # Importing the LLM for NLP tasks

# Specify the path to Tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load a text generation or text classification pipeline from Hugging Face's transformers
llm = pipeline('text-generation', model='gpt-3.5-turbo')  # Example: GPT model for text generation

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image).lower()
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()  # Convert page to image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)  # OCR to extract text
    return text

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text

# Function to extract test results using regex
def extract_test_results(text):
    pattern = r"(creatinine|urea|chloride|sodium|blood urea nitrogen \(bun\)|glomerular filtration rate \(gfr\)|Urine Albumin):?\s*(\d+\.?\d*)"
    matches = re.findall(pattern, text)
    data = [{"Test Name": match[0], "Result": match[1]} for match in matches]
    return pd.DataFrame(data)

# Function to perform a large language model task (e.g., summarizing text)
def llm_task(text):
    result = llm(text, max_length=150)  # Generate a summary or other text outputs based on the input
    return result[0]['generated_text']

# Main function
def main():
    # Ask user for the file path
    file_path = input("Enter the file path (image, PDF, CSV, or Excel): ")

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
    elif file_extension.lower() == '.csv':
        # Extract text from CSV
        print("Processing CSV...")
        text = extract_text_from_csv(file_path)
    elif file_extension.lower() in ['.xls', '.xlsx']:
        # Extract text from Excel
        print("Processing Excel...")
        text = extract_text_from_excel(file_path)
    else:
        print("Unsupported file type.")
        return

    # Display the extracted text
    print("Extracted Text: ")
    print(text)

    # Optional: Use LLM to summarize or process the extracted text
    summarized_text = llm_task(text)
    print("Summarized Text (LLM):")
    print(summarized_text)

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
import openai
import cv2
import pandas as pd
import pytesseract
import re
import fitz  # PyMuPDF for handling PDF files
import os
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set your OpenAI API key here
openai.api_key = 'your-openai-api-key'

# Function to extract text from an image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image).lower()
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()  # Convert page to image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)  # OCR to extract text
    return text

# Function to extract text from CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text

# Function to extract text from Excel
def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = df.to_string().lower()  # Convert all data into a string and make it lowercase
    return text

# Function to extract test results using regex
def extract_test_results(text):
    pattern = r"(creatinine|urea|chloride|sodium|blood urea nitrogen \(bun\)|glomerular filtration rate \(gfr\)|Urine Albumin):?\s*(\d+\.?\d*)"
    matches = re.findall(pattern, text)
    data = [{"Test Name": match[0], "Result": match[1]} for match in matches]
    return pd.DataFrame(data)

# Function to use OpenAI GPT-3.5 for text generation
def llm_task(text):
    response = openai.Completion.create(
        model="text-davinci-003",  # You can use GPT-3.5 model or GPT-4 if available
        prompt=text,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Main function
def main():
    # Ask user for the file path
    file_path = input("Enter the file path (image, PDF, CSV, or Excel): ")

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
    elif file_extension.lower() == '.csv':
        # Extract text from CSV
        print("Processing CSV...")
        text = extract_text_from_csv(file_path)
    elif file_extension.lower() in ['.xls', '.xlsx']:
        # Extract text from Excel
        print("Processing Excel...")
        text = extract_text_from_excel(file_path)
    else:
        print("Unsupported file type.")
        return

    # Display the extracted text
    print("Extracted Text: ")
    print(text)

    # Optional: Use OpenAI GPT-3.5 to summarize or process the extracted text
    summarized_text = llm_task(text)
    print("Summarized Text (LLM):")
    print(summarized_text)

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
    main()'''


