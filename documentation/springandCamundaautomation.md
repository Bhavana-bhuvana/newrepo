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

## Understanding localhost and Configuring It

Spring Boot applications run on `localhost:8080` by default. You can configure it in `application.properties`:
```properties
server.port=9090
```
To make the API accessible on other systems within the same network, use your system's IP address instead of `localhost`.

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


