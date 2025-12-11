# Z-Fighting Debug Plan

## Problem
Z-fighting is still occurring despite multiple fixes. Need to systematically identify root cause.

## Possible Causes

### 1. Identical Depth Values
**Hypothesis**: Triangles are producing identical depth values at shared pixels
**Test**: Log depth values when two triangles compete for same pixel
**Fix**: Ensure depth values are always different

### 2. Depth Buffer Precision
**Hypothesis**: Double precision not sufficient, or depth range too large
**Test**: Check depth value ranges, test with normalized depth
**Fix**: Normalize depth to [0,1] range or use better precision

### 3. Depth Interpolation Issues
**Hypothesis**: Barycentric interpolation producing identical results
**Test**: Compare interpolated depths for adjacent triangles
**Fix**: Use perspective-correct interpolation or add per-triangle offset

### 4. Rendering Order Dependency
**Hypothesis**: Order of triangle rendering affects which wins
**Test**: Render triangles in different orders, check if issue persists
**Fix**: Ensure consistent ordering or use better depth comparison

### 5. Coplanar Triangles Still Sharing Vertices
**Hypothesis**: Despite unique vertices, triangles still share exact positions
**Test**: Check if triangle vertices are truly unique
**Fix**: Add small epsilon offset to shared positions

### 6. Depth Bias Not Applied Correctly
**Hypothesis**: Bias calculation or application is wrong
**Test**: Log bias values and verify they're being applied
**Fix**: Fix bias calculation or increase bias values

## Testing Strategy

### Phase 1: Diagnostic Logging
- Add logging to capture depth values when conflicts occur
- Log triangle indices, depth values, bias values
- Identify patterns in when z-fighting occurs

### Phase 2: Isolate the Issue
- Test with single model (box or pyramid)
- Test with two separate models
- Test with coplanar triangles
- Test with nearly coplanar triangles

### Phase 3: Test Fixes Systematically
- Test each fix independently
- Measure improvement after each fix
- Revert if no improvement

## Implementation Plan

1. Add diagnostic mode to rasterization
2. Log depth conflicts to file
3. Analyze logs to identify pattern
4. Implement targeted fix based on findings
5. Test fix and verify improvement
