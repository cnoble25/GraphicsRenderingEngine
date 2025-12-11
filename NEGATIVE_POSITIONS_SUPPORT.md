# Negative Positions Support

## Overview

The Graphics Rendering Engine **fully supports negative positions** for both objects and lights. All position values use `double` (signed floating-point) types, allowing positions anywhere in 3D space, including negative coordinates.

## Supported Values

- **Object Positions**: Any `double` value (positive, negative, or zero)
- **Light Positions**: Any `double` value (positive, negative, or zero)
- **Scale Values**: Any `double` value (can be negative for mirroring effects)

## Code Verification

### C++ Backend
- `Vec3_API` uses `double x, y, z` (signed)
- `vec3` class uses `double e[3]` (signed)
- No validation restricts positions to positive values
- Transform operations support negative positions

### C# UI
- `Vec3_API` uses `double x, y, z` (signed)
- `PositionX`, `PositionY`, `PositionZ` properties use `double` (signed)
- TextBox bindings accept any numeric value including negatives

## Example Usage

### C++ API
```cpp
SceneObject_API obj;
obj.transform.position.x = -5.0;   // Negative X
obj.transform.position.y = -3.0;   // Negative Y
obj.transform.position.z = -10.0;  // Negative Z
add_object_to_scene(scene, &obj);
```

### C# UI
```csharp
var obj = new SceneObject();
obj.PositionX = -5.0;   // Negative X
obj.PositionY = -3.0;   // Negative Y
obj.PositionZ = -10.0;  // Negative Z
```

## Testing

A test program `test_negative_positions.cpp` verifies:
- Objects can be created with negative positions
- Objects can be updated to negative positions
- Lights can be positioned at negative coordinates
- Mixed positive/negative positions work correctly

To run the test:
```bash
g++ -o test_negative_positions test_negative_positions.cpp -I./src/cpp -L./build-ui -lGraphicsRendererAPI
./test_negative_positions
```

## Notes

- Negative positions are fully supported throughout the rendering pipeline
- No special handling is required - just use negative values as normal
- The UI TextBox controls accept negative numbers (e.g., "-5.0")
- All mathematical operations (transforms, rotations, scaling) work correctly with negative positions

## Fixed Issues

- Fixed `abs()` to `std::abs()` in `vertex.h` for proper floating-point comparison
- Fixed type inconsistency (`float` → `double`) in `vertex.h` for consistency
