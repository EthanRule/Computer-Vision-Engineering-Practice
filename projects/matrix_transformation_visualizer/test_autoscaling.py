import numpy as np
import sys
import os

# Add the current directory to the path so we can import DataEngine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataEngine import DataEngine

def test_scaling_behavior():
    engine = DataEngine()
    
    print("=== Testing Auto-Scaling Functionality ===")
    
    # Test 1: Small values
    print("Test 1: Small values")
    small_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    engine.set_matrix(small_matrix, "small_values")
    original = engine.get_original_shape()
    transformed = engine.get_transformed_shape()
    print(f"Original shape range: X=({min(original[0]):.2f}, {max(original[0]):.2f}), Y=({min(original[1]):.2f}, {max(original[1]):.2f})")
    if transformed is not None:
        print(f"Transformed shape range: X=({min(transformed[0]):.2f}, {max(transformed[0]):.2f}), Y=({min(transformed[1]):.2f}, {max(transformed[1]):.2f})")
    print()
    
    # Test 2: Large scaling transformation
    print("Test 2: Large scaling transformation")
    engine.apply_scaling(10, 5)
    original = engine.get_original_shape()
    transformed = engine.get_transformed_shape()
    print(f"Original shape range: X=({min(original[0]):.2f}, {max(original[0]):.2f}), Y=({min(original[1]):.2f}, {max(original[1]):.2f})")
    if transformed is not None:
        print(f"Transformed shape range: X=({min(transformed[0]):.2f}, {max(transformed[0]):.2f}), Y=({min(transformed[1]):.2f}, {max(transformed[1]):.2f})")
    print()
    
    # Test 3: Large matrix values
    print("Test 3: Large matrix values")
    large_matrix = np.array([[50, -30], [20, 40]])
    engine.set_matrix(large_matrix, "large_values")
    original = engine.get_original_shape()
    transformed = engine.get_transformed_shape()
    print(f"Current matrix: {engine.get_current_matrix()}")
    print(f"Original shape range: X=({min(original[0]):.2f}, {max(original[0]):.2f}), Y=({min(original[1]):.2f}, {max(original[1]):.2f})")
    if transformed is not None:
        print(f"Transformed shape range: X=({min(transformed[0]):.2f}, {max(transformed[0]):.2f}), Y=({min(transformed[1]):.2f}, {max(transformed[1]):.2f})")
    print()
    
    # Test 4: Non-2x2 matrix (should still work for shape visualization)
    print("Test 4: Non-2x2 matrix")
    large_matrix_6x2 = np.array([
        [1, 0],
        [2, 1],  
        [3, 2],
        [-1, 4],
        [0.5, -2],
        [5, 1]
    ])
    engine.set_matrix(large_matrix_6x2, "6x2_matrix")
    print(f"Matrix shape: {engine.get_current_matrix().shape}")
    print(f"Matrix:\n{engine.get_current_matrix()}")
    
    # Visualization will use the original shape since this is not a 2x2 transformation matrix
    original = engine.get_original_shape()
    print(f"Original shape range: X=({min(original[0]):.2f}, {max(original[0]):.2f}), Y=({min(original[1]):.2f}, {max(original[1]):.2f})")

if __name__ == "__main__":
    test_scaling_behavior()
