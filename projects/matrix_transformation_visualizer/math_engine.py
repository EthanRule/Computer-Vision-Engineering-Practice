import numpy as np

class MathEngine:
    """Handles all matrix operations and transformations"""
    
    def __init__(self):
        self.current_matrix = np.eye(3)  # 3x3 identity matrix
        self.history = []
        self.history_index = -1
        
    def set_matrix(self, matrix):
        """Set the current transformation matrix"""
        self.current_matrix = matrix.copy()
        self._save_to_history()
        
    def get_matrix(self):
        """Get the current transformation matrix"""
        return self.current_matrix.copy()
        
    def create_rotation_matrix(self, angle_degrees):
        """Create 3D rotation matrix around Z axis"""
        angle = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])
        
    def create_scaling_matrix(self, sx, sy, sz=1.0):
        """Create 3D scaling matrix"""
        return np.array([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  sz]
        ])
        
    def create_translation_matrix(self, tx, ty, tz=0.0):
        """Create 3D translation matrix"""
        return np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        
    def apply_transformation(self, transform_matrix):
        """Apply transformation to current matrix"""
        self.current_matrix = transform_matrix @ self.current_matrix
        self._save_to_history()
        
    def reset_matrix(self):
        """Reset to identity matrix"""
        self.current_matrix = np.eye(3)
        self._save_to_history()
        
    def _save_to_history(self):
        """Save current state to history"""
        # Remove any future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
            
        self.history.append(self.current_matrix.copy())
        self.history_index += 1
        
        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
            
    def undo(self):
        """Undo last operation"""
        if self.history_index > 0:
            self.history_index -= 1
            self.current_matrix = self.history[self.history_index].copy()
            return True
        return False
        
    def redo(self):
        """Redo last undone operation"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_matrix = self.history[self.history_index].copy()
            return True
        return False
        
    def transform_points(self, points):
        """Transform a set of 3D points using current matrix"""
        if points.shape[1] == 2:
            # Add z=0 for 2D points
            points_3d = np.column_stack([points, np.zeros(len(points))])
        else:
            points_3d = points
            
        # Apply transformation
        transformed = points_3d @ self.current_matrix.T
        return transformed
