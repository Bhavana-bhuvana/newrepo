

### 2. Preparing Camunda Input (Object Creation)
- The function `prepareCamundaInput(String text)` processes extracted text and creates structured JSON objects:
  - Uses **regex** to extract test names and values.
  - Standardizes test names (e.g., `sodium serum` → `sodium`).
  - Creates JSON objects for each test.
  - Groups `sodium` and `potassium` into a single **Electrolytes** object.
  - Returns a `JsonArray` containing test objects.

### Explanation of Object Creation and Conditional Logic for DMN Processing

#### **Object Creation (`prepareCamundaInput` Function)**
1. **Extracting Test Names & Values:**  
   - The function `prepareCamundaInput(String text)` first applies **regular expressions (regex)** to extract test names and corresponding values from the extracted text.  
   - Example: If the input text contains `"Creatinine: 1.2 mg/dL"`, the function will detect `"Creatinine"` as the **test name** and `1.2` as the **value**.

2. **Standardizing Test Names:**  
   - Since different labs may use different formats (e.g., `"Sodium Serum"` vs `"Sodium"`), the function maps variations to a **standard name**.
   - Example:
     ```java
     if (testName.equalsIgnoreCase("sodium serum")) {
         testName = "sodium";
     }
     ```

3. **Creating JSON Objects:**  
   - Each test is stored in a JSON format:
     ```json
     { "variables": { "creatinine": { "value": 1.2, "type": "Double" } } }
     ```
   - These objects are added to a `JsonArray`, which is later processed in the Camunda API.

4. **Handling Electrolytes (Special Case):**  
   - If **Sodium** and **Potassium** exist in the extracted data, they are grouped into a single **Electrolytes object**:
     ```json
     { "variables": { "sodium": { "value": 138, "type": "Double" }, "potassium": { "value": 4.2, "type": "Double" } } }
     ```

---

#### **DMN Processing (`inputCamunda` Function)**
1. **Iterating Over Test Objects:**  
   - The function `inputCamunda(JsonArray camundaInput)` processes the structured test objects by iterating through the list.

2. **Identifying the Right DMN Table for Each Test:**  
   - The function checks the **test name** and selects the correct **Camunda DMN table** from `DMN_KEYS`:
     ```java
     if (testName.equalsIgnoreCase("bun")) {
         dmnKey = "Decision_01t35it";
     } else if (testName.equalsIgnoreCase("creatinine")) {
         dmnKey = "Decision_0bpci3c";
     } else if (testName.equalsIgnoreCase("sodium") || testName.equalsIgnoreCase("potassium")) {
         dmnKey = "Decision_0oywdb5";  // Electrolytes DMN
     }
     ```

3. **Handling Special Cases (Electrolytes Grouping):**  
   - If **both Sodium and Potassium** exist in the dataset, they should be processed **together** using the **Electrolytes DMN table**.
   - The function checks if both are present before making a grouped request:
     ```java
     if (testName.equalsIgnoreCase("sodium") || testName.equalsIgnoreCase("potassium")) {
         if (!processedElectrolytes) { // Ensure it is called only once
             processedElectrolytes = true;
             dmnKey = "Decision_0oywdb5";
             callCamunda(electrolytesObject, dmnKey, allResponses);
         }
     } else {
         callCamunda(testObject, dmnKey, allResponses);
     }
     ```

4. **Calling the Camunda API:**  
   - Finally, the test object is passed to Camunda using:
     ```java
     callCamunda(testObject, dmnKey, allResponses);
     ```

---

### **Summary of Conditional Flow**
| Test Name       | DMN Table Key      | Special Handling? |
|----------------|-------------------|------------------|
| BUN           | `Decision_01t35it` | No |
| Creatinine    | `Decision_0bpci3c` | No |
| Sodium & Potassium | `Decision_0oywdb5` | Processed together as Electrolytes |
| GFR           | `Decision_1aazp67` | No |
| UACR         | `Decision_1h0cvj7` | No |

This logic ensures that each test result is mapped to the correct DMN table while handling **special cases like Electrolytes** efficiently. 🚀
