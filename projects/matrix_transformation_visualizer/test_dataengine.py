import unittest
import numpy as np
import numpy.testing as npt
from DataEngine import DataEngine

class TestDataEngine(unittest.TestCase):
    """Comprehensive test suite for DataEngine class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engine = DataEngine()
        self.tolerance = 1e-10

    def test_initialization(self):
        """Test proper initialization of DataEngine"""
        # Test default values
        npt.assert_array_equal(self.engine.get_current_matrix(), np.eye(2))
        self.assertEqual(len(self.engine.get_history()), 1)
        self.assertEqual(self.engine.get_history()[0], "initial")
        
        # Test original shape initialization
        expected_shape = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        npt.assert_array_equal(self.engine.get_original_shape(), expected_shape)

    # =====================================================
    # NORMAL OPERATION TESTS
    # =====================================================
    
    def test_set_matrix_normal(self):
        """Test setting a normal 2x2 matrix"""
        test_matrix = np.array([[2, 1], [0, 3]])
        result = self.engine.set_matrix(test_matrix, "test_matrix")
        
        npt.assert_array_equal(self.engine.get_current_matrix(), test_matrix)
        self.assertEqual(self.engine.get_history()[-1], "test_matrix")
        self.assertIsNotNone(result)

    def test_basic_matrix_operations(self):
        """Test basic matrix operations with normal inputs"""
        # Test matrix addition
        matrix_a = np.array([[1, 2], [3, 4]])
        matrix_b = np.array([[5, 6], [7, 8]])
        self.engine.set_matrix(matrix_a)
        
        result = self.engine.matrix_addition(matrix_b)
        expected = np.array([[6, 8], [10, 12]])
        npt.assert_array_equal(result, expected)
        
        # Test matrix multiplication
        self.engine.set_matrix(matrix_a)
        result = self.engine.matrix_multiplication(matrix_b)
        expected = matrix_a @ matrix_b
        npt.assert_array_equal(result, expected)
        
        # Test scalar multiplication
        self.engine.set_matrix(matrix_a)
        result = self.engine.scalar_multiplication(2.5)
        expected = matrix_a * 2.5
        npt.assert_array_equal(result, expected)

    def test_transformation_matrices(self):
        """Test creation of transformation matrices"""
        # Test rotation matrix
        angle = 90
        rotation = self.engine.create_rotation_matrix(angle)
        expected = np.array([[0, -1], [1, 0]])
        npt.assert_array_almost_equal(rotation, expected, decimal=10)
        
        # Test scaling matrix
        scale_x, scale_y = 2.0, 3.0
        scaling = self.engine.create_scaling_matrix(scale_x, scale_y)
        expected = np.array([[2.0, 0], [0, 3.0]])
        npt.assert_array_equal(scaling, expected)
        
        # Test shear matrix
        shear_x, shear_y = 0.5, 0.3
        shear = self.engine.create_shear_matrix(shear_x, shear_y)
        expected = np.array([[1, 0.5], [0.3, 1]])
        npt.assert_array_equal(shear, expected)
        
        # Test reflection matrices
        for axis in ['x', 'y', 'xy']:
            reflection = self.engine.create_reflection_matrix(axis)
            self.assertEqual(reflection.shape, (2, 2))

    def test_advanced_matrix_operations(self):
        """Test advanced matrix operations"""
        test_matrix = np.array([[4, 2], [1, 3]])
        self.engine.set_matrix(test_matrix)
        
        # Test determinant
        det = self.engine.determinant()
        expected_det = np.linalg.det(test_matrix)
        self.assertAlmostEqual(det, expected_det, places=10)
        
        # Test inverse
        inverse = self.engine.inverse_matrix()
        expected_inverse = np.linalg.inv(test_matrix)
        npt.assert_array_almost_equal(inverse, expected_inverse)
        
        # Test eigenvalues and eigenvectors
        eigenvals, eigenvecs = self.engine.eigenvalues_eigenvectors()
        expected_vals, expected_vecs = np.linalg.eig(test_matrix)
        npt.assert_array_almost_equal(np.sort(eigenvals), np.sort(expected_vals))
        
        # Test SVD
        U, S, Vt = self.engine.singular_value_decomposition()
        expected_U, expected_S, expected_Vt = np.linalg.svd(test_matrix)
        npt.assert_array_almost_equal(S, expected_S)

    def test_undo_redo_normal(self):
        """Test normal undo/redo operations"""
        initial_matrix = self.engine.get_current_matrix()
        
        # Apply some operations
        self.engine.create_rotation_matrix(45)
        self.engine.create_scaling_matrix(2, 3)
        
        # Test undo
        self.assertTrue(self.engine.can_undo())
        self.assertTrue(self.engine.undo())
        
        # Should be back to rotation
        current = self.engine.get_current_matrix()
        self.assertFalse(np.array_equal(current, initial_matrix))
        
        # Test redo
        self.assertTrue(self.engine.can_redo())
        self.assertTrue(self.engine.redo())

    # =====================================================
    # EDGE CASE TESTS
    # =====================================================
    
    def test_set_matrix_wrong_dimensions(self):
        """Test setting matrix with wrong dimensions"""
        # Test 1x2 matrix
        with self.assertRaises(ValueError):
            self.engine.set_matrix(np.array([[1, 2]]))
        
        # Test 3x3 matrix
        with self.assertRaises(ValueError):
            self.engine.set_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        
        # Test 2x3 matrix
        with self.assertRaises(ValueError):
            self.engine.set_matrix(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_matrix_operations_incompatible_dimensions(self):
        """Test matrix operations with incompatible dimensions"""
        self.engine.set_matrix(np.array([[1, 2], [3, 4]]))
        
        # Test addition with wrong dimensions
        with self.assertRaises(ValueError):
            self.engine.matrix_addition(np.array([[1, 2, 3], [4, 5, 6]]))
        
        # Test multiplication with incompatible dimensions
        with self.assertRaises(ValueError):
            self.engine.matrix_multiplication(np.array([[1], [2], [3]]))

    def test_singular_matrix_operations(self):
        """Test operations on singular (non-invertible) matrices"""
        singular_matrix = np.array([[1, 2], [2, 4]])  # det = 0
        self.engine.set_matrix(singular_matrix)
        
        # Test determinant is zero
        det = self.engine.determinant()
        self.assertAlmostEqual(det, 0, places=10)
        
        # Test inverse returns None for singular matrix
        inverse = self.engine.inverse_matrix()
        self.assertIsNone(inverse)
        
        # Test set_inverse_matrix returns False
        self.assertFalse(self.engine.set_inverse_matrix())

    def test_reflection_invalid_axis(self):
        """Test reflection with invalid axis"""
        with self.assertRaises(ValueError):
            self.engine.create_reflection_matrix("z")
        
        with self.assertRaises(ValueError):
            self.engine.create_reflection_matrix("invalid")

    def test_undo_redo_edge_cases(self):
        """Test undo/redo at boundaries"""
        # Test undo at beginning
        self.assertFalse(self.engine.undo())
        self.assertFalse(self.engine.can_undo())
        
        # Add operation and test redo at end
        self.engine.create_rotation_matrix(30)
        self.assertFalse(self.engine.can_redo())
        self.assertFalse(self.engine.redo())

    def test_dot_product_incompatible_vectors(self):
        """Test dot product with incompatible vector dimensions"""
        vector_a = np.array([1, 2])
        vector_b = np.array([1, 2, 3])
        
        with self.assertRaises(ValueError):
            self.engine.dot_product(vector_a, vector_b)

    def test_empty_history_operations(self):
        """Test operations when history is manipulated"""
        # Clear internal history (edge case)
        self.engine.history = []
        self.engine.current_index = -1
        
        self.assertFalse(self.engine.can_undo())
        self.assertFalse(self.engine.can_redo())

    # =====================================================
    # EXTREME CASE TESTS
    # =====================================================
    
    def test_very_large_matrices(self):
        """Test with matrices containing very large numbers"""
        large_matrix = np.array([[1e10, 2e10], [3e10, 4e10]])
        self.engine.set_matrix(large_matrix)
        
        # Should still work
        det = self.engine.determinant()
        self.assertIsInstance(det, float)
        self.assertTrue(np.isfinite(det))

    def test_very_small_matrices(self):
        """Test with matrices containing very small numbers"""
        small_matrix = np.array([[1e-15, 2e-15], [3e-15, 4e-15]])
        self.engine.set_matrix(small_matrix)
        
        # Should still work
        det = self.engine.determinant()
        self.assertIsInstance(det, float)

    def test_zero_matrix(self):
        """Test operations on zero matrix"""
        zero_matrix = np.zeros((2, 2))
        self.engine.set_matrix(zero_matrix)
        
        # Test determinant is zero
        self.assertEqual(self.engine.determinant(), 0)
        
        # Test inverse is None
        self.assertIsNone(self.engine.inverse_matrix())
        
        # Test trace is zero
        self.assertEqual(self.engine.trace(), 0)
        
        # Test rank is 0
        self.assertEqual(self.engine.matrix_rank(), 0)

    def test_extreme_rotation_angles(self):
        """Test rotation with extreme angles"""
        # Very large angle
        self.engine.create_rotation_matrix(36000)  # 100 full rotations
        result_large = self.engine.get_current_matrix()
        
        # Should be approximately identity due to full rotations
        npt.assert_array_almost_equal(result_large, np.eye(2), decimal=10)
        
        # Negative angle
        self.engine.create_rotation_matrix(-90)
        result_neg = self.engine.get_current_matrix()
        expected_neg = np.array([[0, 1], [-1, 0]])
        npt.assert_array_almost_equal(result_neg, expected_neg, decimal=10)

    def test_extreme_scaling_factors(self):
        """Test scaling with extreme factors"""
        # Very large scaling
        self.engine.create_scaling_matrix(1e6, 1e6)
        large_scale = self.engine.get_current_matrix()
        self.assertEqual(large_scale[0, 0], 1e6)
        self.assertEqual(large_scale[1, 1], 1e6)
        
        # Very small scaling
        self.engine.create_scaling_matrix(1e-6, 1e-6)
        small_scale = self.engine.get_current_matrix()
        self.assertEqual(small_scale[0, 0], 1e-6)
        self.assertEqual(small_scale[1, 1], 1e-6)
        
        # Zero scaling (singular matrix)
        self.engine.create_scaling_matrix(0, 1)
        zero_scale = self.engine.get_current_matrix()
        self.assertEqual(self.engine.determinant(), 0)

    def test_extreme_shear_values(self):
        """Test shear transformation with extreme values"""
        # Very large shear
        self.engine.create_shear_matrix(1000, 1000)
        large_shear = self.engine.get_current_matrix()
        self.assertEqual(large_shear[0, 1], 1000)
        self.assertEqual(large_shear[1, 0], 1000)
        
        # Very small shear
        self.engine.create_shear_matrix(1e-10, 1e-10)
        small_shear = self.engine.get_current_matrix()
        self.assertEqual(small_shear[0, 1], 1e-10)
        self.assertEqual(small_shear[1, 0], 1e-10)

    def test_history_overflow(self):
        """Test behavior when history exceeds maximum size"""
        # Fill history beyond max_history
        original_max = self.engine.max_history
        self.engine.max_history = 5  # Set small limit for testing
        
        # Add more operations than max_history
        for i in range(10):
            self.engine.create_rotation_matrix(i * 10)
        
        # History should be limited to max_history
        self.assertLessEqual(len(self.engine.history), self.engine.max_history)
        
        # Restore original max_history
        self.engine.max_history = original_max

    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits"""
        # Matrix with values near machine epsilon
        epsilon_matrix = np.array([[np.finfo(float).eps, 0], [0, np.finfo(float).eps]])
        self.engine.set_matrix(epsilon_matrix)
        
        # Should still compute determinant
        det = self.engine.determinant()
        self.assertIsInstance(det, float)
        
        # Test with values that might cause overflow
        large_val = np.sqrt(np.finfo(float).max) / 2
        extreme_matrix = np.array([[large_val, 0], [0, large_val]])
        self.engine.set_matrix(extreme_matrix)
        
        det = self.engine.determinant()
        self.assertTrue(np.isfinite(det))

    def test_complex_transformation_chain(self):
        """Test a complex chain of transformations"""
        # Start with identity
        self.engine.reset_to_identity()
        
        # Apply multiple transformations
        transformations = [
            lambda: self.engine.create_rotation_matrix(45),
            lambda: self.engine.create_scaling_matrix(2, 0.5),
            lambda: self.engine.create_shear_matrix(0.3, 0.2),
            lambda: self.engine.create_reflection_matrix('x'),
            lambda: self.engine.scalar_multiplication(1.5),
        ]
        
        for transform in transformations:
            transform()
        
        # Test that we can undo all the way back
        undo_count = 0
        while self.engine.can_undo() and undo_count < 100:  # Safety limit
            self.engine.undo()
            undo_count += 1
        
        # Should be back to initial state (approximately)
        final_matrix = self.engine.get_current_matrix()
        npt.assert_array_almost_equal(final_matrix, np.eye(2), decimal=10)

    def test_matrix_info_comprehensive(self):
        """Test comprehensive matrix information gathering"""
        test_matrix = np.array([[3, 1], [0, 2]])
        self.engine.set_matrix(test_matrix)
        
        info = self.engine.get_matrix_info()
        
        # Test all info fields exist
        required_fields = [
            'matrix', 'determinant', 'trace', 'rank',
            'norm_frobenius', 'norm_2', 'eigenvalues', 'eigenvectors',
            'svd_U', 'svd_S', 'svd_Vt', 'is_invertible',
            'is_symmetric', 'is_orthogonal'
        ]
        
        for field in required_fields:
            self.assertIn(field, info)
        
        # Test specific values
        self.assertEqual(info['determinant'], 6)  # 3*2 - 1*0
        self.assertEqual(info['trace'], 5)  # 3 + 2
        self.assertTrue(info['is_invertible'])
        self.assertFalse(info['is_symmetric'])

    def test_shape_operations(self):
        """Test operations on shapes"""
        # Test custom shape
        custom_shape = np.array([[0, 2, 2, 0], [0, 0, 2, 2]])
        self.engine.set_original_shape(custom_shape)
        
        npt.assert_array_equal(self.engine.get_original_shape(), custom_shape)
        
        # Apply transformation and check result
        self.engine.create_scaling_matrix(2, 3)
        transformed = self.engine.get_transformed_shape()
        expected_transformed = np.array([[0, 4, 4, 0], [0, 0, 6, 6]])
        npt.assert_array_equal(transformed, expected_transformed)

    def test_error_handling_robustness(self):
        """Test error handling in various scenarios"""
        # Test with NaN values
        nan_matrix = np.array([[np.nan, 1], [2, 3]])
        self.engine.set_matrix(nan_matrix)
        
        # Operations should handle NaN gracefully
        det = self.engine.determinant()
        self.assertTrue(np.isnan(det))
        
        # Test with infinity
        inf_matrix = np.array([[np.inf, 1], [2, 3]])
        self.engine.set_matrix(inf_matrix)
        
        det = self.engine.determinant()
        self.assertTrue(np.isinf(det))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
