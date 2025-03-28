# **Flask-Based Web Application Documentation**

## **2. Why Flask?**
Flask is a lightweight and flexible web framework suitable for small to medium-scale applications. It was chosen over Django for this project because:

- **Minimalistic**: Flask provides only essential features, allowing full control over application architecture.
- **Faster Development**: Flask is simple and has minimal setup, making it easier to develop and deploy.
- **Flexibility**: Developers can choose their database, authentication method, and other components.
- **Better for APIs**: Flask is highly suitable for building REST APIs, which is the core requirement of this project.

### **Flask vs. Django**
| Feature | Flask | Django |
|---------|-------|--------|
| Type | Micro-framework | Full-stack framework |
| Flexibility | High, allows custom architecture | Follows predefined structure (MVT) |
| Built-in Features | Minimal, needs extensions | Comes with built-in authentication, admin panel, ORM |
| Learning Curve | Easier for beginners | Steeper due to built-in components |
| Best Use Case | Small to medium applications, APIs | Large applications with complex requirements |

## **3. Framework vs. Library**

### **Framework**
- A framework provides a structured way to develop applications.
- It enforces a particular design pattern and provides built-in tools.
- Example: Flask, Django, Spring Boot.

### **Library**
- A library is a collection of reusable functions and tools that developers can use without following a specific structure.
- It is used to extend functionality but does not enforce a design pattern.
- Example: Pandas (for data processing), OpenCV (for image processing).

## **4. Server vs. Database**

### **Server**
- A server handles client requests, processes them, and returns a response.
- It can be a web server (serving web pages) or an application server (running business logic).
- In this application, Flask acts as a web server, handling HTTP requests and processing file uploads.

### **Database**
- A database stores, retrieves, and manages data.
- In this project, we are not using a database, but if needed, MongoDB or PostgreSQL could be used to store extracted test results.

## **5. How Flask Processes Requests in This Application**
1. **Client Sends Request**:
   - A user uploads a file through a web interface or API request.
   - The request is sent to the Flask server (`/upload` endpoint).

2. **Flask Handles the Request**:
   - Flask processes the request and extracts the uploaded file.
   - The file is saved in the `uploads` folder.

3. **File Type Detection & Processing**:
   - Flask determines the file type (image, PDF, CSV, Excel) and calls the respective extraction function.

4. **Processing and Response**:
   - The extracted text is processed, and test results are identified using regex.
   - The extracted data is returned as a JSON response.

5. **Response Sent to Client**:
   - The JSON response contains the extracted medical test results.
   - The client receives the response and displays it.

## **6. Role of the Server in This Application**
- The Flask server:
  - Listens for incoming HTTP requests (file uploads).
  - Processes uploaded files based on their type.
  - Extracts text and formats it into JSON.
  - Sends a structured JSON response back to the client.

## **7. Future Enhancements**
- **Database Integration**: Store extracted test results in MongoDB or PostgreSQL.
- **User Authentication**: Secure API access with login credentials.
- **Docker Deployment**: Containerize the Flask application for portability.

## **8. Conclusion**
Flask was chosen for its simplicity and efficiency in handling file processing and API development. It acts as a server that receives requests, processes files, and returns results, without the complexity of a full-stack framework like Django. This makes it a perfect choice for this project’s requirements.

Here's a detailed line-by-line explanation of your Flask application, including its implementation, workflow, lifecycle, API structure, and Flask basics.

---

# **Flask-Based File Processing API: A Complete Breakdown**
This Flask-based web application allows users to upload files (images, PDFs, CSVs, and Excel files). It extracts text from the files and attempts to find medical test results using regex-based text extraction.

---

## **1. Understanding Flask Basics**
### **What is Flask?**
Flask is a lightweight and micro web framework for Python used to create web applications. It is simple yet powerful, allowing developers to build REST APIs and web applications quickly.

### **Flask Lifecycle and Workflow**
1. **User Request:** The client (browser/Postman) sends an HTTP request (GET, POST).
2. **Flask App Routes the Request:** Flask checks the URL and matches it to a defined route.
3. **Request Processing:**
   - If the route allows GET, Flask returns an HTML form (for file upload).
   - If the route allows POST, Flask processes the uploaded file.
4. **Response Generation:** Flask processes the data and sends a JSON response with extracted test results.
5. **Response Sent to the User:** The client receives a response and displays the results.

---

## **2. Line-by-Line Explanation**
### **Step 1: Import Required Libraries**
```python
from flask import Flask, request, jsonify
import os
import pandas as pd
import pytesseract as pt
from PIL import Image
import pdfplumber
import re
```
- `Flask`: The core Flask framework for web application development.
- `request`: Handles incoming HTTP requests (e.g., file uploads).
- `jsonify`: Converts Python dictionaries into JSON responses.
- `os`: Used to create directories and manage file paths.
---

###  Initialize Flask Application**
```python
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
```
- `app = Flask(__name__)`: Creates a Flask web application instance.
- `UPLOAD_FOLDER = "uploads"`: Defines the folder where uploaded files will be stored.
- `os.makedirs(UPLOAD_FOLDER, exist_ok=True)`: Ensures the directory exists (creates it if missing).
- `app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER`: Stores the upload directory in Flask's configuration.

---

### **Step 5: Define Flask Routes**
#### **Homepage - File Upload Form**
```python
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
```
- Returns an HTML form where users can upload a file.

#### **File Upload and Processing**
```python
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
```
- Handles `POST` requests for file uploads.
- Saves the uploaded file in the `uploads` folder.

### **Step 6: Run the Flask App**
```python
if __name__ == '__main__':
    app.run(debug=True, port=8000)
```
- Starts the Flask application on `localhost:8000`.

---
### **Understanding `if __name__ == '__main__':` in Flask**
The line:
```python
if __name__ == '__main__':
    app.run(debug=True, port=8000)
#### **Breaking it Down Word by Word**
- `if`: A conditional statement that checks a condition.
- `__name__`: A special Python variable that stores the name of the current module.
- `==`: A comparison operator.
- `'__main__'`: The value assigned to `__name__` when the script is run directly.
- `app.run(debug=True, port=8000)`: Starts the Flask application.

#### **How Flask Starts the Application**
1. When you run the script (`python app.py`), Python assigns `"__main__"` to `__name__`.
2. The `if` condition evaluates to `True`, so `app.run(debug=True, port=8000)` executes.
3. This starts Flask’s built-in development server, which listens for incoming HTTP requests.

---

```
is a standard Python practice used to ensure that the script runs only when executed directly, not when imported as a module.


### **How Flask Creates an API**
Flask creates an API by defining routes using decorators like `@app.route()`. Here’s how it works:

1. **Flask Initialization**
   ```python
   app = Flask(__name__)  
   ```
   - Creates a Flask application instance.
   - `__name__` helps Flask determine where to look for resources (like templates, static files).

2. **Defining API Routes**
   ```python
   @app.route('/upload', methods=['POST'])
   def upload_file():
       # Handle file upload and return a response
   ```
   - `@app.route('/upload', methods=['POST'])` maps the `upload_file()` function to the `/upload` URL.
   - When a `POST` request is sent to `/upload`, Flask executes `upload_file()` and returns a response.

3. **Running the Application**
   ```python
   if __name__ == '__main__':
       app.run(debug=True, port=8000)
   ```
   - Starts the Flask server on `http://127.0.0.1:8000`.

---

### **Flask's Built-in Server: What It Does in API Creation**
Flask comes with a built-in development server, which:
- **Handles Requests**: Listens for HTTP requests and routes them to the correct function.
- **Processes Responses**: Converts Python responses (e.g., JSON) into HTTP responses.
- **Provides Debugging**: With `debug=True`, Flask reloads automatically when code changes.
- **Runs Locally**: By default, it runs on `127.0.0.1` (localhost), meaning it's accessible only on the same machine.

#### **Do We Need a Server to Create an API?**
Yes, an API needs a server to:
- Accept client requests (GET, POST, etc.).
- Process data (like extracting text).
- Send responses back.

Flask's built-in server is **only for development**. In production, use:
- **Gunicorn** (for Linux-based deployment).
- **Waitress** (for Windows).
- **Docker/Kubernetes** (for containerized deployment).

---

### **Where Can We Run This Flask Application?**
You can run the Flask app on:
- **Local Machine**: Using `python app.py`, it runs on `http://127.0.0.1:8000`.
- **Cloud Platforms**:
  - AWS EC2
  - Google Cloud Run
  - Azure App Services
- **Docker Container**: Using a Dockerfile to package and run it in any environment.
- **Production Server**:
  - Nginx + Gunicorn for Linux.
  - IIS + Waitress for Windows.


