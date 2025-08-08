import numpy as np
import sys
import os

# Add the current directory to the path so we can import DataEngine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataEngine import DataEngine

def test_new_functionality():
    engine = DataEngine()
    
    print("=== Testing Matrix Creation ===")
    
    # Test creating different sized matrices
    print("Creating 3x3 identity matrix:")
    identity = engine.create_identity_matrix(3)
    print(identity)
    print()
    
    print("Creating 4x2 random matrix:")
    random_matrix = engine.create_random_matrix(4, 2, -2, 2)
    print(random_matrix)
    print()
    
    print("Creating 6x2 example matrix:")
    example_matrix = np.array([
        [1, 0],
        [0, 1],  
        [2, 3],
        [-1, 2],
        [0.5, -0.5],
        [3, -1]
    ])
    engine.set_matrix(example_matrix, "6x2_example")
    print("6x2 matrix:")
    print(engine.current_matrix)
    print()
    
    print("=== Testing Transformation Chaining ===")
    
    # Test transformation chaining
    print("Applying rotation to 6x2 matrix:")
    try:
        rotated = engine.apply_rotation(45)
        print(rotated)
        print()
    except Exception as e:
        print(f"Rotation failed: {e}")
        print()
    
    print("Then applying scaling:")
    try:
        scaled = engine.apply_scaling(2, 0.5)
        print(scaled)
        print()
    except Exception as e:
        print(f"Scaling failed: {e}")
        print()
    
    print("Matrix dimensions:", engine.current_matrix.shape)
    print()
    
    print("=== Testing Matrix Multiplication ===")
    
    # Create a 2x6 matrix to multiply with our 6x2 matrix
    matrix_2x6 = engine.create_random_matrix(2, 6, -1, 1)
    print("Created 2x6 matrix:")
    print(matrix_2x6)
    print()
    
    # Reset to 6x2 and try matrix multiplication
    engine.set_matrix(example_matrix, "6x2_for_multiplication")
    print("6x2 matrix * 2x6 matrix:")
    try:
        result = engine.matrix_multiply(matrix_2x6)
        print("Result shape:", result.shape)
        print(result)
    except Exception as e:
        print(f"Matrix multiplication failed: {e}")

if __name__ == "__main__":
    test_new_functionality()
