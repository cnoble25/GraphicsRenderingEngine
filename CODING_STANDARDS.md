# Coding Standards for Graphics Rendering Engine

This document outlines the coding standards and style guidelines for the Graphics Rendering Engine project.

## Table of Contents
1. [General Principles](#general-principles)
2. [Naming Conventions](#naming-conventions)
3. [Code Formatting](#code-formatting)
4. [Const Correctness](#const-correctness)
5. [Error Handling](#error-handling)
6. [Memory Management](#memory-management)
7. [Header Organization](#header-organization)
8. [Comments and Documentation](#comments-and-documentation)

---

## General Principles

1. **Clarity over Cleverness**: Write code that is easy to understand
2. **Consistency**: Follow these standards consistently throughout the codebase
3. **Modern C++**: Use C++17 features appropriately
4. **Exception Safety**: Ensure exception safety guarantees where applicable
5. **RAII**: Prefer RAII over manual resource management

---

## Naming Conventions

### Classes
- **PascalCase**: `class Model`, `class CudaBuffer`, `class RendererAPI`
- Use descriptive names that indicate purpose

### Functions and Methods
- **camelCase**: `addObjectToScene()`, `renderScene()`, `copyFromHost()`
- Use verb-noun pattern for actions
- Use descriptive names that indicate what the function does

### Variables
- **camelCase**: `imageWidth`, `triangleCount`, `cudaStatus`
- Use descriptive names
- Avoid abbreviations unless widely understood (e.g., `idx` for index)

### Constants
- **UPPER_SNAKE_CASE**: `MAX_TRIANGLES`, `DEFAULT_LUMINOSITY`, `EPSILON`
- Use `constexpr` when possible

### Private Members
- **Trailing underscore**: `ptr_`, `count_`, `transformed_dirty_`
- Makes it clear when accessing private members

### Enums
- **PascalCase** for enum name: `enum class RendererError`
- **UPPER_SNAKE_CASE** for enum values: `SUCCESS`, `INVALID_SCENE_HANDLE`

### Files
- **snake_case**: `renderer_api.cpp`, `cuda_memory.h`, `ray_trace_cuda.cu`
- Headers: `.h` or `.hpp`
- Source: `.cpp` or `.cu` (for CUDA)

---

## Code Formatting

### Indentation
- Use **4 spaces** (no tabs)
- Configure your editor to show tabs as spaces

### Line Length
- Maximum **100 characters** per line
- Break long lines at logical points

### Braces
- **Attach braces** to the statement (K&R style)
- Always use braces for control structures, even single-line

```cpp
// Good
if (condition) {
    doSomething();
}

// Bad
if (condition)
    doSomething();
```

### Spacing
- One space after keywords: `if (condition)`, `for (int i = 0; i < n; ++i)`
- No space before parentheses in function calls: `functionCall()`
- Space around binary operators: `a + b`, `x = y`
- No space around unary operators: `-x`, `*ptr`

### Pointer and Reference Alignment
- **Left-aligned**: `int* ptr`, `const std::string& str`
- Consistent with LLVM style

---

## Const Correctness

### Use `const` Whenever Possible

1. **Const methods**: Mark methods that don't modify state as `const`
```cpp
// Good
double intersect(const ray& r) const;

// Bad
double intersect(const ray& r);  // Should be const if doesn't modify state
```

2. **Const parameters**: Use `const&` for parameters that aren't modified
```cpp
// Good
void process(const std::vector<vertex>& vertices);

// Bad
void process(std::vector<vertex>& vertices);  // If not modified
```

3. **Const variables**: Mark variables as `const` when not modified
```cpp
// Good
const int imageWidth = 800;
const auto aspectRatio = 16.0 / 9.0;

// Bad
int imageWidth = 800;  // If never modified
```

4. **Const member functions**: Use `[[nodiscard]]` for const methods that return values
```cpp
[[nodiscard]] double magnitude() const;
[[nodiscard]] bool valid() const;
```

---

## Error Handling

### Use Exception-Safe Code
- Prefer exceptions for error handling in C++ code
- Convert exceptions to error codes at C API boundaries
- Use RAII to ensure no resource leaks

### Error Codes
- Use `RendererError` enum for error codes
- Return descriptive error codes, not just 0/1
- Provide error messages via `get_error_message()`

### CUDA Errors
- Use `cuda_utils` namespace functions for CUDA error checking
- Use `CHECK_CUDA()` macro for automatic error checking
- Never ignore CUDA errors

---

## Memory Management

### CUDA Memory
- Always use `CudaBuffer<T>` for CUDA device memory
- Never use manual `cudaMalloc`/`cudaFree`
- Let RAII handle cleanup automatically

### Host Memory
- Prefer `std::vector` over raw arrays
- Use smart pointers when appropriate
- Avoid raw `new`/`delete` in application code

### Move Semantics
- Use move semantics for large objects
- Mark move constructors/assignments as `noexcept`
- Prefer `std::move()` when transferring ownership

---

## Header Organization

### Include Order
1. Corresponding header (if .cpp file)
2. System headers (`<iostream>`, `<vector>`, etc.)
3. Third-party headers (`<cuda_runtime.h>`, etc.)
4. Project headers (`"vec3.h"`, `"model.h"`, etc.)

### Include Guards
- Use `#pragma once` for new headers
- Or use traditional include guards:
```cpp
#ifndef HEADER_NAME_H
#define HEADER_NAME_H
// ...
#endif // HEADER_NAME_H
```

### Forward Declarations
- Use forward declarations when possible to reduce compile time
- Include headers only when definitions are needed

---

## Comments and Documentation

### File Headers
- Include author and creation date for new files
- Brief description of file purpose

### Function Documentation
- Document public API functions
- Explain parameters and return values
- Note any exceptions that may be thrown
- Document preconditions/postconditions when important

### Inline Comments
- Explain **why**, not **what**
- Keep comments up-to-date with code
- Remove commented-out code (use version control instead)

### Example
```cpp
/**
 * Renders the scene to a PPM image file
 * @param scene Scene handle (must be valid)
 * @param output_path Path to output file
 * @param image_width Width of output image in pixels
 * @param image_height Height of output image in pixels
 * @param luminosity Light intensity multiplier
 * @param render_mode Rendering mode (currently only ray tracing supported)
 * @return Non-zero on success, zero on failure
 * @throws std::runtime_error on CUDA errors
 */
int render_scene(SceneHandle scene, const char* output_path, 
                 int image_width, int image_height, 
                 double luminosity, RenderMode_API render_mode);
```

---

## Modern C++ Features

### Use When Appropriate
- **`auto`**: For type deduction when type is obvious
- **`constexpr`**: For compile-time constants
- **`noexcept`**: For functions that never throw
- **`[[nodiscard]]`**: For functions whose return values should not be ignored
- **Range-based for**: `for (const auto& item : container)`
- **Smart pointers**: `std::unique_ptr`, `std::shared_ptr` when appropriate

### Avoid
- Raw pointers for ownership (use smart pointers or RAII)
- C-style casts (use `static_cast`, `const_cast`, etc.)
- `goto` statements (use RAII and exceptions)
- Manual memory management (use RAII)

---

## Code Review Checklist

Before submitting code, ensure:
- [ ] Follows naming conventions
- [ ] Proper const correctness
- [ ] No memory leaks (use RAII)
- [ ] Error handling is appropriate
- [ ] Comments explain non-obvious code
- [ ] Code compiles without warnings
- [ ] No `goto` statements
- [ ] No manual CUDA memory management
- [ ] Headers have include guards
- [ ] Code is formatted consistently

---

## Tools

### Formatting
- Use `clang-format` with `.clang-format` configuration
- Run before committing: `clang-format -i src/**/*.{cpp,h,cu}`

### Linting
- Use compiler warnings: `-Wall -Wextra -Wpedantic`
- Address all warnings

### Static Analysis
- Consider using tools like `cppcheck` or `clang-tidy`

---

## Examples

### Good Code
```cpp
class Model {
public:
    [[nodiscard]] double intersect(const ray& r) const {
        double min_t = -1.0;
        for (const auto& vertex : transformed_vertices) {
            const double t = vertex.ray_intersection(r);
            if (t > 0 && (min_t < 0 || t < min_t)) {
                min_t = t;
            }
        }
        return min_t;
    }
    
private:
    std::vector<vertex> transformed_vertices_;
};
```

### Bad Code
```cpp
class Model {
public:
    double intersect(ray& r) {  // Should be const
        double min = -1;
        for (vertex i : transformed_vertices) {  // Should use const&
            double t = i.ray_intersection(r);
            if (t > 0) {
                if (min == -1) {
                    min = t;
                }
                if (t < min) {  // Redundant check
                    min = t;
                }
            }
        }
        return min;
    }
    
private:
    std::vector<vertex> transformed_vertices;  // Should have trailing underscore
};
```

---

## Questions?

If you're unsure about a style decision, refer to:
1. This document
2. Existing codebase patterns
3. C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/
