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
 ---
 ### **What is PythonAnywhere?**
PythonAnywhere is a cloud-based Python development and hosting environment that allows you to:
- Run Python scripts online without installing anything.
- Host Flask/Django web apps.
- Access files and databases remotely.
- Schedule automated tasks.

**Why Use PythonAnywhere?**
- No need to set up servers manually.
- Free-tier available for basic projects.
- Publicly accessible URLs for Flask apps.

#### **Deploying a Flask App on PythonAnywhere**
1. **Create an account** at [https://www.pythonanywhere.com](https://www.pythonanywhere.com).
2. **Upload your Flask app** (`app.py` and dependencies).
3. **Set up a web app** under the "Web" tab.
4. **Configure the WSGI file** to point to `app.py`.
5. **Reload the web app**, and your API will be publicly available.

---
### **PythonAnywhere Workflow and WSGI Configuration Explained**  

PythonAnywhere is a cloud-based platform that allows you to host and run Python applications (Flask, Django, etc.) without needing a dedicated server.  

---

## **1️⃣ Workflow of PythonAnywhere**  
### **Step 1: Setting Up Your Web App**  
- You create a **Flask** (or Django) app and upload the files to PythonAnywhere.  
- The source code is stored in the **home directory** (e.g., `/home/Bhavanakomal/mysite`).  
- PythonAnywhere provides a **built-in server** to run your application.

### **Step 2: Virtual Environment**  
- A virtual environment (`/home/Bhavanakomal/.virtualenvs/flaskk`) helps keep dependencies isolated from the system Python installation.
- If your app needs specific versions of libraries (e.g., Flask, NumPy), you install them inside this virtual environment.

### **Step 3: WSGI Configuration (The Connection to the Server)**  
- The **WSGI configuration file (`bhavanakomal_pythonanywhere_com_wsgi.py`)** is the main entry point for your web application.
- It **links your Flask/Django app to the PythonAnywhere web server**.
- When someone visits your site, the **WSGI server reads this file** and runs your application.

### **Step 4: Running the Web App**  
- When your app starts, the **server executes the WSGI script**, loads the application, and serves responses.
- If you make changes, you **must reload the web app** for the new code to take effect.

---

## **2️⃣ Why Do We Need the WSGI Configuration File?**  
WSGI (**Web Server Gateway Interface**) is the standard interface between Python web applications (Flask, Django) and the server.

Your **WSGI file (`bhavanakomal_pythonanywhere_com_wsgi.py`)** does these things:
1. **Tells the server how to run your Flask app.**  
   - It loads your Python application from `/home/Bhavanakomal/mysite`.  
2. **Specifies which Python version to use.**  
   - It ensures your app runs with Python 3.9 instead of other versions.  
3. **Handles incoming requests and responses.**  
   - It passes requests to Flask and sends responses back to the client.  

---

## **3️⃣ What is the Role of the Built-in Server?**
- PythonAnywhere provides a **pre-configured web server** that handles traffic for your app.
- You **do not need to run `flask run` manually**; the server automatically starts when your app is deployed.
- The built-in server is **not for production**; for large-scale apps, you should use **Gunicorn, uWSGI, or Nginx**.

---

## **4️⃣ Why Use PythonAnywhere Instead of Local Hosting?**  
| Feature            | PythonAnywhere                          | Local Hosting (Own Server) |
|--------------------|--------------------------------|---------------------------|
| **Ease of Use**    | No setup needed; pre-configured | Must install Flask, Nginx, etc. |
| **Accessibility**  | Accessible from anywhere | Limited to local network (unless port forwarded) |
| **Maintenance**    | Automatic updates, backups | Must handle updates manually |
| **Security**       | Secured by PythonAnywhere | Must configure firewalls, SSL |

---

## **5️⃣ Summary**  
- **PythonAnywhere hosts your Flask/Django app without needing a dedicated server.**  
- **The WSGI file (`bhavanakomal_pythonanywhere_com_wsgi.py`) connects your app to the PythonAnywhere server.**  
- **It ensures the right Python version (3.9) and dependencies are used.**  
- **The built-in server handles requests, eliminating the need to run Flask manually.**  


PythonAnywhere is a cloud-based platform that allows you to host and run Python applications (Flask, Django, etc.) without needing a dedicated server.  

---

## **1️⃣ Workflow of PythonAnywhere**  
### **Step 1: Setting Up Your Web App**  
- You create a **Flask** (or Django) app and upload the files to PythonAnywhere.  
- The source code is stored in the **home directory** (e.g., `/home/Bhavanakomal/mysite`).  
- PythonAnywhere provides a **built-in server** to run your application.

### **Step 2: Virtual Environment**  
- A virtual environment (`/home/Bhavanakomal/.virtualenvs/flaskk`) helps keep dependencies isolated from the system Python installation.
- If your app needs specific versions of libraries (e.g., Flask, NumPy), you install them inside this virtual environment.

### **Step 3: WSGI Configuration (The Connection to the Server)**  
- The **WSGI configuration file (`bhavanakomal_pythonanywhere_com_wsgi.py`)** is the main entry point for your web application.
- It **links your Flask/Django app to the PythonAnywhere web server**.
- When someone visits your site, the **WSGI server reads this file** and runs your application.

### **Step 4: Running the Web App**  
- When your app starts, the **server executes the WSGI script**, loads the application, and serves responses.
- If you make changes, you **must reload the web app** for the new code to take effect.

---

## **2️⃣ Why Do We Need the WSGI Configuration File?**  
WSGI (**Web Server Gateway Interface**) is the standard interface between Python web applications (Flask, Django) and the server.

Your **WSGI file (`bhavanakomal_pythonanywhere_com_wsgi.py`)** does these things:
1. **Tells the server how to run your Flask app.**  
   - It loads your Python application from `/home/Bhavanakomal/mysite`.  
2. **Specifies which Python version to use.**  
   - It ensures your app runs with Python 3.9 instead of other versions.  
3. **Handles incoming requests and responses.**  
   - It passes requests to Flask and sends responses back to the client.  

---

## **3️⃣ What is the Role of the Built-in Server?**
- PythonAnywhere provides a **pre-configured web server** that handles traffic for your app.
- You **do not need to run `flask run` manually**; the server automatically starts when your app is deployed.
- The built-in server is **not for production**; for large-scale apps, you should use **Gunicorn, uWSGI, or Nginx**.

---

## **4️⃣ Why Use PythonAnywhere Instead of Local Hosting?**  
| Feature            | PythonAnywhere                          | Local Hosting (Own Server) |
|--------------------|--------------------------------|---------------------------|
| **Ease of Use**    | No setup needed; pre-configured | Must install Flask, Nginx, etc. |
| **Accessibility**  | Accessible from anywhere | Limited to local network (unless port forwarded) |
| **Maintenance**    | Automatic updates, backups | Must handle updates manually |
| **Security**       | Secured by PythonAnywhere | Must configure firewalls, SSL |

---

## **5️⃣ Summary**  
- **PythonAnywhere hosts your Flask/Django app without needing a dedicated server.**  
- **The WSGI file (`bhavanakomal_pythonanywhere_com_wsgi.py`) connects your app to the PythonAnywhere server.**  
- **It ensures the right Python version (3.9) and dependencies are used.**  
- **The built-in server handles requests, eliminating the need to run Flask manually.**  

Would you like me to show the content of your **WSGI file** and explain it in detail? 🚀

### **Making Your Flask App Public Using CMD**
If you're running Flask on your local machine, by default, it runs on `127.0.0.1:8000`, which is **only accessible from your own computer**. To make it public, you have two simple ways:

#### **Method 1: Use Flask's Host Setting**
Instead of:
```python
app.run(debug=True, port=8000)
```
Run:
```python
app.run(host='0.0.0.0', port=8000)
```
- `0.0.0.0` makes it accessible to other devices on the same network.
- Your local IP (e.g., `192.168.1.100:8000`) can be used by others on your Wi-Fi.

#### **Method 2: Expose Flask API Publicly Using `ngrok`**
1. **Install ngrok**:
   - Windows: Download from [ngrok.com](https://ngrok.com/download) and extract.
   - Linux/Mac: Install via terminal:
     ```
     curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
     echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && \
     sudo apt update && sudo apt install ngrok
     ```
2. **Run Flask on port 8000**:
   ```
   python app.py
   ```
3. **Start ngrok**:
   ```
   ngrok http 8000
   ```


  




