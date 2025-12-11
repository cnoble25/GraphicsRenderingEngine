# Z-Fighting Testing Plan

## Current Hypothesis
The z-fighting is likely caused by **identical depth values** being produced for different triangles at the same pixel location.

## Test Cases to Execute

### Test 1: Verify Depth Values Are Unique
**Action**: Add micro-offset per triangle (DONE)
**Expected**: Each triangle should have slightly different depth
**Status**: Implemented - triangle_micro_offset added

### Test 2: Check Depth Range
**Action**: Log min/max depth values in scene
**Expected**: Depth range should be reasonable (not too large/small)
**Status**: TODO

### Test 3: Verify Bias Is Applied
**Action**: Log bias values and verify they're non-zero
**Expected**: All triangles should have positive bias
**Status**: TODO

### Test 4: Test Rendering Order
**Action**: Render triangles in reverse order
**Expected**: If order-dependent, issue should change
**Status**: TODO

### Test 5: Check for Exact Vertex Duplication
**Action**: Verify unique vertices are truly unique
**Expected**: No two triangles should share exact same vertex positions
**Status**: TODO

## Next Steps
1. Build and test with micro-offset
2. If still failing, enable debug logging
3. Analyze depth conflicts
4. Implement targeted fix
