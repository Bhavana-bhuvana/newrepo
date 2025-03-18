			DOCUMENTATION for goo.py
1.Text extraction from image ,pdf ,csv and excel formats 

________________________________________
It uses Optical Character Recognition (OCR) technology and regular expressions (regex) to identify and extract specific data points.
Code Functionality and Libraries
1.	Libraries Used: 
o	cv2 (OpenCV):
Used for image processing, particularly for converting images to grayscale and applying thresholding. This enhances the image for better OCR accuracy.
o	pytesseract:
A Python wrapper for Google's Tesseract OCR engine. It is used to extract text from preprocessed images.
o	fitz (PyMuPDF):
Handles PDF files by extracting pages and converting them into images, which can then be processed for OCR.
o	Pandas:
Provides tools for data manipulation and storage. In this script, it is used to structure the extracted test results into a tabular format (DataFrame).
o	re (Regular Expressions):
Facilitates pattern matching to identify and extract specific medical test results from the extracted text.
o	PIL (Pillow):
Used for image manipulation, particularly to convert PDF-rendered images to grayscale for preprocessing.

________________________________________
Code Features
1.	Image Preprocessing:
o	Converts the image to grayscale and applies binary thresholding to improve the clarity of text, making it easier for OCR to recognize.
2.	Text Extraction:
o	For images: Uses OpenCV and Tesseract to read text.
o	For PDFs: Converts each page into an image using PyMuPDF, preprocesses it, and applies OCR.
3.	Data Extraction:
o	Uses regex to identify specific medical test names and their corresponding values from the extracted text.
4.	Output:
o	Displays the extracted test results and saves them as a CSV file for further use.
________________________________________
Here’s a simplified explanation of each function in the code:
Function Explanations
1.	preprocess_image(image)
o	Converts the input image to grayscale and applies binary thresholding to enhance text visibility for OCR.
2.	extract_text_from_image(image_path)
o	Reads an image from the given file path, preprocesses it, and extracts text using Tesseract OCR.
3.	extract_text_from_pdf(pdf_path)
o	Opens a PDF, converts each page to an image, preprocesses the image, and uses Tesseract OCR to extract text.
4.	extract_test_results(text)
o	Scans the given text with a regex pattern to find medical test names and their results, storing them in a structured format (DataFrame).
5.	main()
o	Acts as the entry point: asks for a file path, determines whether it's an image or PDF, processes it, extracts test results, and saves them as a CSV file.
o	me) from the extracted test results for easy manipulation and export.
Important built-in functions:
1. cv2.imread(image_path):
   - Reads an image from the specified file path (image_path) and returns it as a NumPy array. This array represents the pixel data of the image.
2. pytesseract.image_to_string(image):
   - Performs Optical Character Recognition (OCR) on the given image to extract text and returns the extracted text as a string.
3. fitz.open(pdf_path):
   - Opens a PDF file at the specified path (pdf_path) and creates a Document object, which allows access to its pages for further processing.
4. document.load_page(page_num):
   - Loads a specific page (page_num) from the Document object, enabling operations like rendering the page as an image.

5. page.get_pixmap():
   - Converts a loaded PDF page into a pixel-based image (rasterization), which can be processed by image manipulation libraries.6. Image.frombytes("RGB", [pix.width, pix.height], pix.samples):
   - Creates a PIL (Python Imaging Library) image object from raw pixel data, with the specified color mode (RGB), dimensions, and pixel samples.
7. re.findall(pattern, text):
   - Searches the string (text) for all occurrences that match the given regular expression pattern (pattern) and returns them as a list of tuples or strings.
8. pd.DataFrame(data):
   - Creates a Pandas DataFrame from the provided data, which can be a list of dictionaries, a NumPy array, or other compatible formats.

9. files.upload():
   - Allows users to upload files in a Google Colab environment. It returns a dictionary where keys are the file names and values are the corresponding file data.
10. os.path.splitext(image_path):
   - Splits the file path into two parts: the base name and the file extension. Useful for determining the file type.

11. df.to_csv("filename.csv", index=False):
   - Exports the Pandas DataFrame (df) to a CSV file with the specified name (filename.csv). The index=False option excludes the DataFrame's index column from the output.

Each of these functions serves a specific purpose in the workflow, helping to handle file input/output, image processing, text extraction, and data analysis:
1. list(uploaded.keys())[0]:
     - uploaded.keys() :retrieves the keys of the uploaded dictionary, which are the names of the files uploaded using files.upload().
     - list(uploaded.keys()) :converts the dictionary keys into a list.
     - [0] accesses the first element of the list, which is the name of the first uploaded file. 
   - *Purpose:*  
     This extracts the file name of the uploaded file, assuming that only one file is being processed in this case.

2. os.path.splitext(image_path):[1].lower()* 
     - os.path.splitext(image_path) :splits the file name stored in image_path into two parts:
       1. The base name (e.g., "file_name").
       2. The file extension (e.g., ".jpg", ".pdf").
     - [1] accesses the second part of the tuple, which is the file extension (e.g., .jpg or .pdf).
     - .lower() converts the extension to lowercase, ensuring consistent comparison (to handle cases where extensions might be in uppercase, like .JPG or .PDF).
   - Purpose: 
     This determines the type of file (image or PDF) by extracting and normalizing its extension. The code later uses this to decide whether to process the file as an image or a PDF.
How They Work Together:
1. list(uploaded.keys())[0] identifies the name of the uploaded file.
2. os.path.splitext(image_path)[1].lower() extracts the file extension from the name, helping determine the appropriate processing steps.


Current Challenges
1.	Text Extraction from PDFs:
o	Some PDFs might contain text in vectorized format instead of rasterized images, making OCR redundant. Direct text extraction from such PDFs using PyMuPDF's page.get_text() can be a more efficient alternative.
2.	Accuracy of OCR:
o	OCR performance is highly dependent on the quality of the input image. Poor image resolution, noise, or non-standard fonts can lead to errors in text recognition.
o	The pre-processing techniques (thresholding and grayscale conversion) may need optimization for different types of images or PDFs.
3.	Regex Limitations:
o	The current regex patterns are hardcoded and might miss test results with unexpected formats or variations in names.
4.	Error Handling:
o	The script currently lacks detailed error handling for cases like corrupt files, unsupported file types, or missing Tesseract installation.

________________________________________


