"""
Benchmark module for image_operations.py
Tests performance of various image operations with different image sizes and scenarios.
"""

import time
import statistics
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from image_operations import ImageOperations
import os
from typing import Dict, List, Tuple
import json

class ImageOperationsBenchmark:
    def __init__(self):
        self.results = {}
        self.test_images = self._create_test_images()
        
    def _create_test_images(self) -> Dict[str, np.ndarray]:
        """Create test images of different sizes for benchmarking."""
        test_images = {}
        
        # Small image (640x480)
        test_images['small'] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Medium image (1280x720)
        test_images['medium'] = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Large image (1920x1080)
        test_images['large'] = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Very large image (4K - 3840x2160)
        test_images['xlarge'] = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        
        return test_images
    
    def _time_operation(self, operation_func, iterations: int = 10) -> Dict[str, float]:
        """Time an operation multiple times and return statistics."""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            operation_func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'times': times
        }
    
    def benchmark_grayscale(self, image_size: str, iterations: int = 10) -> Dict[str, float]:
        """Benchmark grayscale conversion operation."""
        print(f"Benchmarking grayscale conversion on {image_size} image...")
        
        # Create a temporary ImageOperations instance with test image
        ops = ImageOperations()
        original_img = ops.img.copy()
        ops.img = self.test_images[image_size].copy()
        
        def grayscale_op():
            ops.img = self.test_images[image_size].copy()
            ops.gray_scale()
        
        result = self._time_operation(grayscale_op, iterations)
        
        # Restore original image
        ops.img = original_img
        
        return result
    
    def benchmark_crop(self, image_size: str, iterations: int = 10) -> Dict[str, float]:
        """Benchmark crop operation."""
        print(f"Benchmarking crop operation on {image_size} image...")
        
        ops = ImageOperations()
        original_img = ops.img.copy()
        test_img = self.test_images[image_size]
        
        # Define crop region (middle 50% of image)
        h, w = test_img.shape[:2]
        crop_h_start, crop_h_end = h // 4, 3 * h // 4
        crop_w_start, crop_w_end = w // 4, 3 * w // 4
        
        def crop_op():
            ops.img = test_img.copy()
            ops.crop(crop_h_start, crop_h_end, crop_w_start, crop_w_end)
        
        result = self._time_operation(crop_op, iterations)
        
        # Restore original image
        ops.img = original_img
        
        return result
    
    def benchmark_scale(self, image_size: str, scale_factors: List[int] = [50, 75, 150, 200], iterations: int = 10) -> Dict[int, Dict[str, float]]:
        """Benchmark scaling operation with different scale factors."""
        print(f"Benchmarking scale operation on {image_size} image...")
        
        results = {}
        ops = ImageOperations()
        original_img = ops.img.copy()
        
        for scale_factor in scale_factors:
            print(f"  Testing scale factor: {scale_factor}%")
            
            def scale_op():
                ops.img = self.test_images[image_size].copy()
                ops.scale(scale_factor)
            
            results[scale_factor] = self._time_operation(scale_op, iterations)
        
        # Restore original image
        ops.img = original_img
        
        return results
    
    def benchmark_undo_redo(self, iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark undo/redo operations."""
        print("Benchmarking undo/redo operations...")
        
        ops = ImageOperations()
        original_img = ops.img.copy()
        
        # Setup: perform several operations to have undo history
        ops.gray_scale()
        ops.scale(50)
        ops.crop(100, 400, 100, 400)
        
        def undo_op():
            ops.undo()
        
        def redo_op():
            ops.redo()
        
        undo_results = self._time_operation(undo_op, iterations)
        
        # Reset for redo test
        for _ in range(10):
            ops.undo()
        
        redo_results = self._time_operation(redo_op, iterations)
        
        # Restore original image
        ops.img = original_img
        
        return {
            'undo': undo_results,
            'redo': redo_results
        }
    
    def run_comprehensive_benchmark(self, iterations: int = 10) -> Dict:
        """Run all benchmarks and return comprehensive results."""
        print("=" * 60)
        print("Running Comprehensive Image Operations Benchmark")
        print("=" * 60)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'iterations': iterations,
            'operations': {}
        }
        
        image_sizes = ['small', 'medium', 'large', 'xlarge']
        
        # Benchmark each operation for each image size
        for size in image_sizes:
            print(f"\nTesting {size} image ({self.test_images[size].shape})")
            print("-" * 40)
            
            results['operations'][size] = {}
            
            # Grayscale benchmark
            results['operations'][size]['grayscale'] = self.benchmark_grayscale(size, iterations)
            
            # Crop benchmark
            results['operations'][size]['crop'] = self.benchmark_crop(size, iterations)
            
            # Scale benchmark
            results['operations'][size]['scale'] = self.benchmark_scale(size, iterations=iterations)
        
        # Undo/Redo benchmark (size-independent)
        print("\nTesting undo/redo operations")
        print("-" * 40)
        results['operations']['undo_redo'] = self.benchmark_undo_redo(iterations * 10)
        
        self.results = results
        return results
    
    def print_results(self):
        """Print benchmark results in a readable format."""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Iterations per test: {self.results['iterations']}")
        
        # Print results for each image size
        for size, operations in self.results['operations'].items():
            if size == 'undo_redo':
                continue
                
            image_shape = self.test_images[size].shape
            print(f"\n{size.upper()} IMAGE ({image_shape[1]}x{image_shape[0]} pixels)")
            print("-" * 50)
            
            # Grayscale results
            if 'grayscale' in operations:
                gs = operations['grayscale']
                print(f"Grayscale:     {gs['mean']:.4f}s ± {gs['std_dev']:.4f}s (avg)")
            
            # Crop results
            if 'crop' in operations:
                crop = operations['crop']
                print(f"Crop:          {crop['mean']:.4f}s ± {crop['std_dev']:.4f}s (avg)")
            
            # Scale results
            if 'scale' in operations:
                print("Scale operations:")
                for scale_factor, scale_data in operations['scale'].items():
                    print(f"  {scale_factor}%:        {scale_data['mean']:.4f}s ± {scale_data['std_dev']:.4f}s (avg)")
        
        # Undo/Redo results
        if 'undo_redo' in self.results['operations']:
            print(f"\nUNDO/REDO OPERATIONS")
            print("-" * 50)
            undo_redo = self.results['operations']['undo_redo']
            print(f"Undo:          {undo_redo['undo']['mean']:.6f}s ± {undo_redo['undo']['std_dev']:.6f}s (avg)")
            print(f"Redo:          {undo_redo['redo']['mean']:.6f}s ± {undo_redo['redo']['std_dev']:.6f}s (avg)")
    
    def plot_results(self):
        """Create visualization plots of benchmark results."""
        if not self.results:
            print("No benchmark results to plot.")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Operations Performance Benchmark', fontsize=16)
        
        sizes = ['small', 'medium', 'large', 'xlarge']
        size_labels = ['640x480', '1280x720', '1920x1080', '3840x2160']
        
        # Plot 1: Grayscale performance
        grayscale_times = [self.results['operations'][size]['grayscale']['mean'] for size in sizes]
        ax1.bar(size_labels, grayscale_times, color='skyblue')
        ax1.set_title('Grayscale Conversion Performance')
        ax1.set_ylabel('Time (seconds)')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Crop performance
        crop_times = [self.results['operations'][size]['crop']['mean'] for size in sizes]
        ax2.bar(size_labels, crop_times, color='lightgreen')
        ax2.set_title('Crop Operation Performance')
        ax2.set_ylabel('Time (seconds)')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Scale performance (using 50% scale as example)
        scale_times = [self.results['operations'][size]['scale'][50]['mean'] for size in sizes]
        ax3.bar(size_labels, scale_times, color='orange')
        ax3.set_title('Scale Operation Performance (50%)')
        ax3.set_ylabel('Time (seconds)')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Undo/Redo performance
        undo_redo = self.results['operations']['undo_redo']
        operations = ['Undo', 'Redo']
        times = [undo_redo['undo']['mean'], undo_redo['redo']['mean']]
        ax4.bar(operations, times, color=['red', 'blue'])
        ax4.set_title('Undo/Redo Performance')
        ax4.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save plot
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # timestamp = time.strftime('%Y%m%d_%H%M%S')
        # plot_path = os.path.join(script_dir, f'benchmark_plot_{timestamp}.png')
        # print(f"Benchmark plot saved to: {plot_path}")
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run benchmarks."""
    benchmark = ImageOperationsBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(iterations=5)
    
    # Print results
    benchmark.print_results()
    
    # Create plots
    benchmark.plot_results()

if __name__ == "__main__":
    main()
