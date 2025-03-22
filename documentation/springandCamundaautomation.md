# Spring Boot API Documentation

## Introduction

This project is a Spring Boot application that integrates with Camunda for decision-making. It processes uploaded medical test reports, extracts relevant data, and invokes Camunda Decision Model Notation (DMN) tables to generate insights.

## Table of Contents

1. [What is Spring?](#what-is-spring)
2. [What is Spring Boot?](#what-is-spring-boot)
3. [Understanding Annotations in Spring Boot](#understanding-annotations-in-spring-boot)
4. [Creating Routes in Spring Boot](#creating-routes-in-spring-boot)
5. [Understanding localhost and Configuring It](#understanding-localhost-and-configuring-it)
6. [API Lifecycle and Workflow](#api-lifecycle-and-workflow)
7. [How SpringApplication.run() Starts the Application](#how-springapplicationrun-starts-the-application)
8. [Complete Program Workflow](#complete-program-workflow)
9. [Event Workflow of This Program](#event-workflow-of-this-program)
10. [Postman API Testing Guide](#postman-api-testing-guide)
11. [Accessing the API](#accessing-the-api)

---

## What is Spring?

Spring is a powerful Java framework that provides a comprehensive infrastructure for enterprise applications. It simplifies Java development by providing features like dependency injection, aspect-oriented programming, and transaction management. Spring is widely used for building scalable, secure, and efficient Java applications.

### How Spring Helps in Projects:

- **Modular Architecture**: Enables better organization of code.
- **Dependency Injection**: Simplifies object management and reduces tight coupling.
- **Transaction Management**: Ensures data consistency and reliability.
- **Security Features**: Provides built-in authentication and authorization mechanisms.
- **Integration Capabilities**: Easily integrates with databases, messaging systems, and third-party APIs.

---

## What is Spring Boot?

Spring Boot is a Java-based framework built on top of Spring to simplify application development. It provides an easy way to create standalone, production-ready applications with minimal configuration. Spring Boot comes with embedded servers (Tomcat, Jetty, or Undertow) and auto-configuration features that reduce the need for manual setup.

### Why Use Spring Boot?
- **Eliminates Boilerplate Code**: Reduces XML configurations and simplifies development.
- **Embedded Server**: Comes with a built-in Tomcat, making deployment easier.
- **Microservices Ready**: Ideal for building REST APIs and microservices architectures.
- **Spring Boot Starter Packs**: Provides pre-configured dependencies for common functionalities.
- **Production-Ready Features**: Includes monitoring, health checks, and logging.

---

## Understanding Annotations in Spring Boot

Annotations are used in Spring Boot to reduce configuration complexity. They provide metadata to the Spring framework to manage application components and their behaviors.

### **Why Are Annotations Used?**
- Simplifies configuration by reducing XML-based setup.
- Helps in defining components like controllers, services, and repositories.
- Provides better readability and maintainability.

### **Can We Write Programs Without Annotations?**
Yes, but it would require more manual configuration using XML or Java-based configuration, making the development process complex and time-consuming.

---

## Creating Routes in Spring Boot

Routes in Spring Boot are defined using controller classes and annotations. The **DispatcherServlet** processes incoming requests and maps them to the appropriate methods.

### **How Does It Work?**

Spring Boot uses the **Spring MVC framework**, which has a **DispatcherServlet** at its core. When a request is received:

1. The **DispatcherServlet** intercepts the request.
2. It checks the **URL pattern** and matches it with the controller methods.
3. If a matching method is found, it executes the method and returns the response.

### **Example:**
```java
@RestController
@RequestMapping("/api")
public class HelloWorldController {
    @GetMapping("/hello")
    public String sayHello() {
        return "Hello, World!";
    }
}
```
### **Explanation:**
1. **`@RestController`**: Marks this class as a RESTful web service.
2. **`@RequestMapping("/api")`**: Sets the base URL for the API.
3. **`@GetMapping("/hello")`**: Creates an endpoint `GET /api/hello` that returns "Hello, World!".

### **How Annotations Help in Creating Routes**

Annotations play a crucial role in defining routes in Spring Boot:

1. **`@RequestMapping("/")`** defines the **base path** for all methods in the class.
2. **`@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping`** directly define routes without requiring additional XML configurations.
3. The **DispatcherServlet** automatically maps these annotations to actual URL paths.

---
### Understanding `localhost` and Configuring It in Spring Boot Applications

#### What is `localhost`?

`localhost` is a special hostname that refers to the current device or machine. It resolves to the IP address `127.0.0.1` in IPv4 or `::1` in IPv6. This is commonly used to access services running on the same machine. 

When you run a Spring Boot application or any web application on your local computer, it will listen on `localhost` by default. This means that the application can only be accessed from the same machine it is running on.

For example, when a Spring Boot application runs on `localhost:8080`, it means the application is listening for HTTP requests on port 8080 of the local machine. To access this service, you would open a browser and navigate to:
```
http://localhost:8080
```

#### How Does Localhost Work?

1. **Binding to Localhost**:
   - When you start a Spring Boot application (or any server-based application), it listens on a specific IP address and port.
   - By default, Spring Boot binds the application to `localhost` (i.e., `127.0.0.1`), meaning the application is accessible only from the local machine.

2. **Accessing Services**:
   - If a Spring Boot app runs on `localhost:8080`, you can only access it from the same machine by using the URL `http://localhost:8080`. If you try to access `localhost:8080` from another machine, the connection will fail because `localhost` refers specifically to the local machine.

#### How to Change the Port and Host in Spring Boot?

In Spring Boot, you can easily configure the server to listen on a different port and even bind it to a specific IP address (instead of `localhost`).

1. **Changing the Port**:
   - By default, Spring Boot runs on port 8080. You can change this in the `application.properties` or `application.yml` file.
   - To change the port to `9090`, for example, you can add the following property:
     ```properties
     server.port=9090
     ```

2. **Binding to a Specific IP Address**:
   - To make the application accessible from other devices on the same network, you can bind the application to your machine's IP address instead of `localhost`.
   - In `application.properties`, you can specify the host and port like this:
     ```properties
     server.address=0.0.0.0
     server.port=9090
     ```
   - `0.0.0.0` means the application will listen on all available network interfaces, making it accessible from any device within the same network (local area network).

   - Alternatively, you can bind it to a specific IP address of your system (for example, `192.168.1.100`):
     ```properties
     server.address=192.168.1.100
     server.port=9090
     ```

   - Once this is configured, other devices in the same network can access your Spring Boot application by typing your system's IP address and the specified port into their browsers:
     ```
     http://192.168.1.100:9090
     ```

#### How Does Accessing APIs and Services Work in Spring Boot?

In Spring Boot, accessing APIs or web services is done using URLs that consist of the following parts:

1. **Protocol**: The communication protocol to use (usually `http` or `https`).
2. **Host**: The hostname or IP address where the service is running (e.g., `localhost`, `192.168.1.100`, or a domain name like `api.example.com`).
3. **Port**: The port number where the service is listening (default HTTP port is 80, default Spring Boot port is 8080).
4. **Path**: The specific API endpoint (e.g., `/api/users`).
5. **Query Parameters**: Optional parameters that can be added to the URL to pass data to the API (e.g., `?id=123`).

For example, if your Spring Boot application runs on `localhost:9090`, and you have a REST API that provides user data at the endpoint `/users`, you can access it with the following URL:
```
http://localhost:9090/users
```

If you were to make this application accessible to others in your network, and your machine's IP address is `192.168.1.100`, the URL for others would look like:
```
http://192.168.1.100:9090/users
```

#### How Does This Relate to Camunda and Other Frameworks?

In a system like **Camunda** (a BPMN process engine) integrated with Spring Boot, **localhost** works the same way.

- Camunda, like Spring Boot, may expose APIs or web endpoints for managing business processes.
- By default, these APIs are accessible on `localhost` and would be available at `http://localhost:8080/engine-rest/...` or a similar URL.
- If you want Camunda's REST API to be accessible across the network, you would configure the Spring Boot application to listen on `0.0.0.0` or a specific IP address, just like any other Spring Boot application.
  
  Example in `application.properties` for Camunda:
  ```properties
  server.address=0.0.0.0
  server.port=8080
  ```

- Once configured, users can interact with Camunda's APIs remotely via URLs such as:
  ```
  http://192.168.1.100:8080/engine-rest/engine
  ```

#### Example Workflow of Accessing an API in Spring Boot

1. **Starting the Application**: 
   - When you run your Spring Boot application, it listens for HTTP requests on a specified host and port (e.g., `localhost:8080`).
   
2. **Making a Request**:
   - A user (or a system) sends an HTTP request to your application by entering a URL in a browser or using an HTTP client like Postman or `curl`. The request may look like:
     ```
     GET http://localhost:8080/api/hello
     ```

3. **Spring Boot Processes the Request**:
   - Spring Boot routes the request to the appropriate controller and handler method based on the URL and HTTP method (GET, POST, PUT, DELETE).
   - The controller processes the request and returns a response, usually in the form of JSON or HTML.

4. **Sending the Response**:
   - The response is sent back to the requester with a status code (e.g., 200 OK for successful requests, 404 Not Found for unknown endpoints).

5. **Accessing the Response**:
   - If the application is running on `localhost`, you can access it from the same machine.
   - If the application is made available on the network (using the system's IP address), others in the same network can access the same URL.

#### Example of Changing the Spring Boot Configuration

```properties
# application.properties

# Make the application accessible on the network
server.address=0.0.0.0
server.port=8080
```

Once you apply this change, you can access the application on any machine in the same local network by using your IP address:
```
http://192.168.1.100:8080
```

### Conclusion

Understanding how `localhost` works is crucial when setting up your Spring Boot application and APIs. By default, Spring Boot binds to `localhost`, making the application only accessible from the same machine. You can change this by modifying the `server.address` and `server.port` properties in `application.properties` to allow other systems within the network to access your application.

Once configured, the application can be accessed from other devices using the machine’s IP address and port number. Whether you're working with Camunda, Spring Boot, or any other framework, the basic concept of localhost and IP-based addressing remains the same.
---

## API Lifecycle and Workflow
1. **Client Request**: A request is sent from a browser, Postman, or another client.
2. **DispatcherServlet**: Intercepts the request.
3. **Handler Mapping**: Finds the correct controller method.
4. **Controller Execution**: Processes the request.
5. **Service & Repository Layers**: Fetch and process data.
6. **Response Handling**: Sends back the response to the client.

---

## How `SpringApplication.run()` Starts the Application
1. Initializes Spring components.
2. Loads configuration files.
3. Starts an embedded server.
4. Launches the application and makes endpoints available.

---

## Complete Program Workflow
- User uploads a medical report.
- Spring Boot extracts test data.
- Data is sent to Camunda for decision processing.
- Processed insights are returned as JSON.

---

## Event Workflow of This Program
- File Upload → Data Extraction → API Call → Decision Processing → Response Sent

---

## Postman API Testing Guide
- Use Postman to test GET, POST, PUT, DELETE requests.
- Example: `POST http://localhost:8080/upload` with file attachment.

---

## Accessing the API

### 1. Using a Browser
- Only **GET** requests can be tested in a browser.
- Example: `http://localhost:8080/api/hello`

### 2. Using Postman
- Allows testing **GET, POST, PUT, DELETE** requests.
- Set headers and request body as needed.

### 3. Using Command Line (cURL)
Run the following command to test a **POST** request:
```sh
curl -X POST http://localhost:8080/upload -F "file=@path/to/file"
```

### 4. Accessing from Another System
1. Find your **local IP address** using:
   ```sh
   ipconfig (Windows) or ifconfig (Mac/Linux)
   ```
2. Run the Spring Boot application.
3. Replace `localhost` with your IP in the URL:
   ```sh
   http://192.168.x.x:8080/upload
   ```
4. Ensure firewall settings allow external access.

---


