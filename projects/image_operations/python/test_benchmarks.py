"""
Quick test script to demonstrate the benchmarking functionality
of the enhanced ImageOperations class.
"""

from image_operations import ImageOperations
import time

def test_basic_operations():
    """Test basic operations with timing."""
    print("Testing ImageOperations with built-in timing...")
    
    # Create an instance
    ops = ImageOperations()
    
    # Enable verbose timing to see real-time performance
    ops.enable_verbose_timing()
    
    print("\nPerforming operations...")
    
    # Test grayscale conversion
    ops.gray_scale()
    
    # Test scaling
    ops.scale(50)
    ops.scale(150)
    ops.scale(75)
    
    # Test cropping
    h, w = ops.img.shape[:2]
    ops.crop(h//4, 3*h//4, w//4, 3*w//4)
    
    # Test undo/redo
    ops.undo()
    ops.redo()
    ops.undo()
    
    # Disable verbose timing
    ops.disable_verbose_timing()
    
    # Print performance summary
    ops.print_performance_summary()
    
    # Export timing data
    ops.export_timing_data("test_timing_results.json")

def stress_test():
    """Stress test operations for better performance data."""
    print("\n\nRunning stress test...")
    
    ops = ImageOperations()
    
    # Perform multiple operations quickly
    print("Performing 10 grayscale conversions...")
    for i in range(10):
        ops.gray_scale()
    
    print("Performing various scale operations...")
    scale_factors = [50, 75, 125, 150, 200, 25, 300, 100]
    for factor in scale_factors:
        ops.scale(factor)
    
    print("Performing undo/redo operations...")
    for i in range(20):
        if i % 2 == 0:
            ops.undo()
        else:
            ops.redo()
    
    # Print final summary
    ops.print_performance_summary()

if __name__ == "__main__":
    test_basic_operations()
    stress_test()
    
    print("\n\nTo run comprehensive benchmarks with multiple image sizes,")
    print("execute: python benchmark.py")
