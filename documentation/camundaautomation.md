# Camunda-Based Test Processing API

This project integrates **Spring Boot** with **Camunda BPM** to process medical test results from uploaded files. The workflow extracts text, evaluates test values using Camunda Decision Model and Notation (DMN) tables, and returns a final score with insights.

## API Workflow

### 1. File Upload and Text Extraction
- **Endpoint:** `POST /upload`
- **Process:**
  - The user uploads a file (PDF, Excel, CSV, or Image).
  - The `TextExtractionService` extracts test names and values.
  - Data is converted into **Camunda JSON format**.
  
### 2. DMN Table Processing with Camunda
- The API determines which **DMN table** should process each test:
  - **BUN** → `Decision_01t35it`
  - **Creatinine** → `Decision_0bpci3c`
  - **Electrolytes (Sodium & Potassium combined)** → `Decision_0oywdb5`
  - **GFR** → `Decision_1aazp67`
  - **UACR** → `Decision_1h0cvj7`
  
- Each test is evaluated separately except for `sodium` and `potassium`, which are processed together under **Electrolytes DMN**.

### Explanation of URL Formation

The URL used in your project for calling Camunda's REST API is:

```
POST http://localhost:8080/engine-rest/decision-definition/key/{dmnKey}/evaluate
```

#### Breaking It Down:
1. **`http://localhost:8080/`**  
   - `http://` → Specifies the HTTP protocol.
   - `localhost` → Refers to the machine where Camunda is running. This means your API is currently accessible only from your system.
   - `8080` → The port where Camunda is running. If needed, this can be changed in Camunda's configuration.

2. **`engine-rest/`**  
   - This is the base path for Camunda's REST API. It allows external applications to interact with Camunda using HTTP requests.

3. **`decision-definition/key/{dmnKey}/evaluate`**  
   - `decision-definition` → Specifies that we are accessing decision models (DMN tables).
   - `key/{dmnKey}` → `{dmnKey}` is a placeholder that will be replaced by the actual key of the DMN table you want to evaluate (e.g., `Decision_01t35it`).
   - `evaluate` → This endpoint is used to send input data to the DMN table for processing.

---

### Customization for Your Use
You can modify the URL based on your requirements:

1. **Change `localhost` to a remote server**  
   If you want to allow other systems to access Camunda, replace `localhost` with your system's IP or a domain name:
   ```
   POST http://192.168.1.100:8080/engine-rest/decision-definition/key/{dmnKey}/evaluate
   ```
   - If deploying on the cloud, replace with a public URL (e.g., `http://yourdomain.com:8080/`).

2. **Change the port number**  
   If another service is running on port 8080, you can configure Camunda to use a different port:
   - Update Camunda's `server.port` in its configuration.
   - Change your API URL accordingly:
     ```
     POST http://localhost:9090/engine-rest/decision-definition/key/{dmnKey}/evaluate
     ```

3. **Use HTTPS for security**  
   If deploying in production, switch to HTTPS:
   ```
   POST https://yourdomain.com/engine-rest/decision-definition/key/{dmnKey}/evaluate
   ```
   - You need an SSL certificate for this.

Would you like me to guide you through setting up a custom domain or securing your API?
- The response includes:
  - **Risk Category** (e.g., Low, Medium, High)
  - **Score** (used for final scoring)

### 4. Final Result Calculation
- Insights are generated for each test.
- The **final score** is calculated as:
  ```
  Final Score = (Sum of all test scores) / (Number of tests)
  ```
- JSON response includes:
  ```json
  {
    "final_score": 3.5,
    "insights": [
      { "test_name": "Creatinine", "risk_category": "High", "score": 5 },
      { "test_name": "BUN", "risk_category": "Medium", "score": 3 }
    ]
  }
  ```

## Starting Camunda BPM
1. **Ensure Camunda is installed** at `C:\camunda\camunda-bpm-run-7.13.0`
2. **Start Camunda using Command Prompt:**
   ```batch
   cd C:\camunda\camunda-bpm-run-7.13.0
   start.bat
   ```
3. **Wait for Camunda to initialize** (approx. 5 seconds)
4. **Verify Camunda is running**:
   - Open `http://localhost:8080/` in a browser.
   - Check if the REST API is accessible.

## Key Components

### `HelloWorldController.java`
- Handles file upload (`/upload`).
- Calls `TextExtractionService` to extract test results.
- Sends extracted data to `CamundaService`.
- Collects responses and generates final insights.

### `CamundaService.java`
- **Starts Camunda BPM** if not running.
- **Calls Camunda REST API** to execute DMN tables.
- **Processes responses** to extract risk categories and scores.
- **Generates final JSON output** combining insights and scores.

## Running the Project
1. **Start Camunda manually or let the API start it automatically.**
2. **Run the Spring Boot application**:
   ```bash
   mvn spring-boot:run
   ```
3. **Upload a test file** using Postman or a frontend.
4. **Receive insights and final scores** in JSON format.

## Future Enhancements
- Improve error handling for invalid inputs.
- Support more test types dynamically.
- Deploy Camunda and API as a cloud service.

---
**Author:** Your Name  
**Project:** Camunda-Based Medical Test Processing

