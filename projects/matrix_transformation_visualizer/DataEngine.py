import numpy as np
from typing import List, Tuple, Optional, Union
import copy

class DataEngine:
    """
    A comprehensive matrix operations engine with undo/redo functionality.
    Supports basic matrix operations, linear transformations, and advanced decompositions.
    """
    
    def __init__(self):
        self.history: List[dict] = []
        self.current_index: int = -1
        self.max_history: int = 50
        
        # Default data
        self.current_matrix = np.eye(2)
        self.original_shape = np.array([[0, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 0]])
        self.transformed_shape = None
        
        # Save initial state
        self._save_state("initial")
    
    def _save_state(self, operation: str, **kwargs):
        """Save current state to history for undo/redo functionality"""
        state = {
            'operation': operation,
            'matrix': copy.deepcopy(self.current_matrix),
            'original_shape': copy.deepcopy(self.original_shape),
            'transformed_shape': copy.deepcopy(self.transformed_shape),
            'timestamp': np.datetime64('now'),
            **kwargs
        }
        
        # Remove future history if we're not at the end
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(state)
        self.current_index += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            self.current_index = len(self.history) - 1
    
    def _restore_state(self, state: dict):
        """Restore engine to a previous state"""
        self.current_matrix = copy.deepcopy(state['matrix'])
        self.original_shape = copy.deepcopy(state['original_shape'])
        self.transformed_shape = copy.deepcopy(state['transformed_shape'])
    
    def undo(self) -> bool:
        """Undo the last operation. Returns True if successful."""
        if self.current_index > 0:
            self.current_index -= 1
            self._restore_state(self.history[self.current_index])
            return True
        return False
    
    def redo(self) -> bool:
        """Redo the next operation. Returns True if successful."""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self._restore_state(self.history[self.current_index])
            return True
        return False
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_index < len(self.history) - 1
    
    def get_history(self) -> List[str]:
        """Get list of operations in history"""
        return [state['operation'] for state in self.history]
    
    def create_identity_matrix(self, size: int) -> np.ndarray:
        """Create an identity matrix of given size"""
        identity = np.eye(size)
        self.set_matrix(identity, f"identity_{size}x{size}")
        return identity
    
    def create_random_matrix(self, rows: int, cols: int, min_val: float = -10, max_val: float = 10) -> np.ndarray:
        """Create a random matrix with specified dimensions"""
        random_matrix = np.random.uniform(min_val, max_val, (rows, cols))
        self.set_matrix(random_matrix, f"random_{rows}x{cols}")
        return random_matrix
    
    def create_zero_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Create a zero matrix with specified dimensions"""
        zero_matrix = np.zeros((rows, cols))
        self.set_matrix(zero_matrix, f"zero_{rows}x{cols}")
        return zero_matrix
    
    def create_ones_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Create a matrix filled with ones"""
        ones_matrix = np.ones((rows, cols))
        self.set_matrix(ones_matrix, f"ones_{rows}x{cols}")
        return ones_matrix
    
    def set_matrix(self, matrix: np.ndarray, operation_name: str = "set_matrix"):
        """Set the current transformation matrix"""
        self.current_matrix = matrix.copy()
        self.transformed_shape = self.apply_transformation()
        self._save_state(operation_name, matrix=matrix)
        return self.transformed_shape
    
    def matrix_addition(self, matrix_b: np.ndarray) -> np.ndarray:
        """Add two matrices"""
        if self.current_matrix.shape != matrix_b.shape:
            raise ValueError("Matrices must have the same shape")
        
        result = self.current_matrix + matrix_b
        self.current_matrix = result
        self.transformed_shape = self.apply_transformation()
        self._save_state("matrix_addition", added_matrix=matrix_b.copy())
        return result
    
    def matrix_multiply(self, matrix_b: np.ndarray) -> np.ndarray:
        """Multiply current matrix by matrix_b"""
        if self.current_matrix.shape[1] != matrix_b.shape[0]:
            raise ValueError(f"Cannot multiply {self.current_matrix.shape} matrix by {matrix_b.shape} matrix: incompatible dimensions")
        
        result = self.current_matrix @ matrix_b
        self.current_matrix = result
        self.transformed_shape = self.apply_transformation()
        self._save_state("matrix_multiplication", multiplied_by=matrix_b.copy())
        return result

    def scalar_multiplication(self, scalar: float) -> np.ndarray:
        """Multiply matrix by a scalar"""
        result = self.current_matrix * scalar
        self.current_matrix = result
        self.transformed_shape = self.apply_transformation()
        self._save_state("scalar_multiplication", scalar=scalar)
        return result
    
    # =====================================================
    # LINEAR TRANSFORMATIONS
    # =====================================================
    
    def apply_transformation(self, shape: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply current transformation matrix to a shape (for 2D visualizations)"""
        target_shape = shape if shape is not None else self.original_shape
        
        # Only apply transformation if current matrix is 2x2 (for visualization)
        if self.current_matrix.shape == (2, 2):
            return self.current_matrix @ target_shape
        else:
            # For non-2x2 matrices, just return the original shape for visualization
            return target_shape
    
    def dot_product(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate dot product of two vectors"""
        if vector_a.shape != vector_b.shape:
            raise ValueError("Vectors must have the same shape")
        return np.dot(vector_a, vector_b)
    
    def create_rotation_matrix(self, angle_degrees: float) -> np.ndarray:
        """Create a 2x2 rotation transformation matrix"""
        angle_rad = np.radians(angle_degrees)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                   [np.sin(angle_rad), np.cos(angle_rad)]])
        self.set_matrix(rotation_matrix, f"rotation_{angle_degrees}deg")
        return rotation_matrix
    
    def apply_rotation(self, angle_degrees: float) -> np.ndarray:
        """Apply rotation transformation to current matrix"""
        rotation_matrix = np.array([[np.cos(np.radians(angle_degrees)), -np.sin(np.radians(angle_degrees))],
                                   [np.sin(np.radians(angle_degrees)), np.cos(np.radians(angle_degrees))]])
        
        # For matrix transformations, multiply current matrix by transformation matrix
        if self.current_matrix.shape[1] == 2:  # Can multiply by 2x2 rotation
            result = self.current_matrix @ rotation_matrix
            self.set_matrix(result, f"applied_rotation_{angle_degrees}deg")
            return result
        else:
            raise ValueError(f"Cannot apply 2x2 rotation to matrix with {self.current_matrix.shape[1]} columns")
    
    def create_scaling_matrix(self, scale_x: float, scale_y: float) -> np.ndarray:
        """Create a 2x2 scaling transformation matrix"""
        scaling_matrix = np.array([[scale_x, 0],
                                  [0, scale_y]])
        self.set_matrix(scaling_matrix, f"scaling_{scale_x}x{scale_y}")
        return scaling_matrix
    
    def apply_scaling(self, scale_x: float, scale_y: float) -> np.ndarray:
        """Apply scaling transformation to current matrix"""
        scaling_matrix = np.array([[scale_x, 0],
                                  [0, scale_y]])
        
        # For matrix transformations, multiply current matrix by transformation matrix
        if self.current_matrix.shape[1] == 2:  # Can multiply by 2x2 scaling
            result = self.current_matrix @ scaling_matrix
            self.set_matrix(result, f"applied_scaling_{scale_x}x{scale_y}")
            return result
        else:
            raise ValueError(f"Cannot apply 2x2 scaling to matrix with {self.current_matrix.shape[1]} columns")
    
    def create_shear_matrix(self, shear_x: float = 0, shear_y: float = 0) -> np.ndarray:
        """Create a 2x2 shear transformation matrix"""
        shear_matrix = np.array([[1, shear_x],
                                [shear_y, 1]])
        self.set_matrix(shear_matrix, f"shear_x{shear_x}_y{shear_y}")
        return shear_matrix
    
    def apply_shear(self, shear_x: float = 0, shear_y: float = 0) -> np.ndarray:
        """Apply shear transformation to current matrix"""
        shear_matrix = np.array([[1, shear_x],
                                [shear_y, 1]])
        
        # For matrix transformations, multiply current matrix by transformation matrix
        if self.current_matrix.shape[1] == 2:  # Can multiply by 2x2 shear
            result = self.current_matrix @ shear_matrix
            self.set_matrix(result, f"applied_shear_x{shear_x}_y{shear_y}")
            return result
        else:
            raise ValueError(f"Cannot apply 2x2 shear to matrix with {self.current_matrix.shape[1]} columns")
    
    def create_reflection_matrix(self, axis: str = "x") -> np.ndarray:
        """Create a reflection transformation matrix"""
        if axis.lower() == "x":
            reflection_matrix = np.array([[1, 0], [0, -1]])
        elif axis.lower() == "y":
            reflection_matrix = np.array([[-1, 0], [0, 1]])
        elif axis.lower() == "xy" or axis.lower() == "origin":
            reflection_matrix = np.array([[-1, 0], [0, -1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'xy'")
        
        self.set_matrix(reflection_matrix, f"reflection_{axis}")
        return reflection_matrix
    
    # =====================================================
    # ADVANCED MATRIX OPERATIONS
    # =====================================================
    
    def determinant(self, matrix: Optional[np.ndarray] = None) -> float:
        """Calculate the determinant of a matrix (only for square matrices)"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        if target_matrix.shape[0] != target_matrix.shape[1]:
            raise ValueError("Determinant is only defined for square matrices")
        return np.linalg.det(target_matrix)
    
    def inverse_matrix(self, matrix: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Calculate the inverse of a matrix"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        
        try:
            det = self.determinant(target_matrix)
            if abs(det) < 1e-10:
                raise np.linalg.LinAlgError("Matrix is singular (determinant is zero)")
            
            inverse = np.linalg.inv(target_matrix)
            return inverse
        except np.linalg.LinAlgError as e:
            print(f"Cannot compute inverse: {e}")
            return None
    
    def set_inverse_matrix(self) -> bool:
        """Set current matrix to its inverse"""
        inverse = self.inverse_matrix()
        if inverse is not None:
            self.set_matrix(inverse, "inverse_matrix")
            return True
        return False
    
    def eigenvalues_eigenvectors(self, matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        eigenvalues, eigenvectors = np.linalg.eig(target_matrix)
        return eigenvalues, eigenvectors
    
    def singular_value_decomposition(self, matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Singular Value Decomposition (SVD)
        Returns U, S, V^T where matrix = U * S * V^T
        """
        target_matrix = matrix if matrix is not None else self.current_matrix
        U, S, Vt = np.linalg.svd(target_matrix)
        return U, S, Vt
    
    def matrix_rank(self, matrix: Optional[np.ndarray] = None) -> int:
        """Calculate the rank of a matrix"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        return np.linalg.matrix_rank(target_matrix)
    
    def matrix_norm(self, matrix: Optional[np.ndarray] = None, norm_ord: Optional[Union[str, int]] = None) -> float:
        """Calculate various norms of a matrix"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        return np.linalg.norm(target_matrix, ord=norm_ord)
    
    def trace(self, matrix: Optional[np.ndarray] = None) -> float:
        """Calculate the trace (sum of diagonal elements) of a matrix (only for square matrices)"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        if target_matrix.shape[0] != target_matrix.shape[1]:
            raise ValueError("Trace is only defined for square matrices")
        return np.trace(target_matrix)
    
    def transpose(self, matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate the transpose of a matrix"""
        target_matrix = matrix if matrix is not None else self.current_matrix
        return target_matrix.T
    
    def set_transpose_matrix(self):
        """Set current matrix to its transpose"""
        transposed = self.transpose()
        self.set_matrix(transposed, "transpose")
    
    # =====================================================
    # UTILITY METHODS
    # =====================================================
    
    def reset_to_identity(self):
        """Reset transformation matrix to identity"""
        self.set_matrix(np.eye(2), "reset_identity")
    
    def get_current_matrix(self) -> np.ndarray:
        """Get the current transformation matrix"""
        return self.current_matrix.copy()
    
    def get_original_shape(self) -> np.ndarray:
        """Get the original shape"""
        return self.original_shape.copy()
    
    def get_transformed_shape(self) -> Optional[np.ndarray]:
        """Get the transformed shape"""
        return self.transformed_shape.copy() if self.transformed_shape is not None else None
    
    def set_original_shape(self, shape: np.ndarray):
        """Set a new original shape"""
        self.original_shape = shape.copy()
        self.transformed_shape = self.apply_transformation()
        self._save_state("set_original_shape", shape=shape.copy())
    
    def get_matrix_info(self) -> dict:
        """Get comprehensive information about the current matrix"""
        matrix = self.current_matrix
        eigenvals, eigenvecs = self.eigenvalues_eigenvectors()
        U, S, Vt = self.singular_value_decomposition()
        
        info = {
            'matrix': matrix,
            'determinant': self.determinant(),
            'trace': self.trace(),
            'rank': self.matrix_rank(),
            'norm_frobenius': self.matrix_norm(norm_ord='fro'),
            'norm_2': self.matrix_norm(norm_ord=2),
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'svd_U': U,
            'svd_S': S,
            'svd_Vt': Vt,
            'is_invertible': abs(self.determinant()) > 1e-10,
            'is_symmetric': np.allclose(matrix, matrix.T),
            'is_orthogonal': np.allclose(matrix @ matrix.T, np.eye(matrix.shape[0]))
        }
        
        return info
    
    def __str__(self) -> str:
        """String representation of the current state"""
        return f"DataEngine - Matrix:\n{self.current_matrix}\nHistory: {len(self.history)} operations"
