package org.example;

import org.bson.Document;
import org.example.sercice.MongoService;
import org.springframework.http.ResponseEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/users")
public class ProcessController {
    @Autowired
    private MongoService mongoService;

    //  Insert User with Dynamic Test Results (NO EMAIL)
    @PostMapping("/add")
    public ResponseEntity<String> addUser(@RequestBody Map<String, Object> requestData) {
        try {
            String name = (String) requestData.get("name");
            Integer age = (Integer) requestData.get("age");
            List<Map<String, Object>> testResults = (List<Map<String, Object>>) requestData.get("testResults");

            mongoService.insertUser(name, age, testResults); // ✅ Removed email
            return ResponseEntity.ok("User added successfully with test results!");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error: " + e.getMessage());
        }
    }

    //  Get All Users
    @GetMapping("/all")
    public ResponseEntity<List<Document>> getAllUsers() {
        return ResponseEntity.ok(mongoService.getAllUsers());
    }

    //  Get User by Name
    @GetMapping("/{name}")
    public ResponseEntity<?> getUserByName(@PathVariable String name) {
        Document user = mongoService.getUserByName(name);
        return (user != null) ? ResponseEntity.ok(user) : ResponseEntity.badRequest().body("User not found!");
    }

    //  Update User Age by Name
    @PutMapping("/update/{name}")
    public ResponseEntity<String> updateUser(@PathVariable String name, @RequestParam int newAge) {
        boolean updated = mongoService.updateUserAge(name, newAge);
        return updated ? ResponseEntity.ok("User updated!") : ResponseEntity.badRequest().body("User not found!");
    }

    //  Delete a User by Name
    @DeleteMapping("/delete/{name}")
    public ResponseEntity<String> deleteUser(@PathVariable String name) {
        boolean deleted = mongoService.deleteUser(name);
        return deleted ? ResponseEntity.ok("User deleted!") : ResponseEntity.badRequest().body("User not found!");
    }
}
