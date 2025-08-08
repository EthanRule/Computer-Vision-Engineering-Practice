import numpy as np
import sys
import os

# Add the current directory to the path so we can import DataEngine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataEngine import DataEngine

def test_transformation_behavior():
    engine = DataEngine()
    
    # Start with a simple matrix
    test_matrix = np.array([[2, 1], [0, 3]])
    engine.set_matrix(test_matrix, "initial_test_matrix")
    print("Initial matrix:")
    print(engine.current_matrix)
    print()
    
    # Apply rotation
    print("After rotation (45 degrees):")
    engine.create_rotation_matrix(45)
    print(engine.current_matrix)
    print()
    
    # Reset and try scaling
    engine.set_matrix(test_matrix, "reset_for_scaling")
    print("Reset to initial matrix:")
    print(engine.current_matrix)
    print()
    
    print("After scaling (2, 3):")
    engine.create_scaling_matrix(2, 3)
    print(engine.current_matrix)
    print()
    
    # Reset and try shear
    engine.set_matrix(test_matrix, "reset_for_shear")
    print("Reset to initial matrix:")
    print(engine.current_matrix)
    print()
    
    print("After shear (1, 0.5):")
    engine.create_shear_matrix(1, 0.5)
    print(engine.current_matrix)
    print()
    
    # Now test inverse on the shear result
    print("After applying inverse to shear matrix:")
    success = engine.set_inverse_matrix()
    if success:
        print(engine.current_matrix)
    else:
        print("Inverse failed!")
    print()

if __name__ == "__main__":
    test_transformation_behavior()
