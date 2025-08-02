# Image Operations Benchmarking

This project now includes comprehensive benchmarking capabilities for measuring the performance of image operations.

## Features Added

### 1. Built-in Timing in ImageOperations Class

The `ImageOperations` class now automatically tracks the execution time of all operations:

- **Automatic timing**: All major operations (grayscale, crop, scale, undo, redo) are automatically timed
- **Performance statistics**: Get detailed stats including average, min, max execution times
- **Verbose mode**: Option to see timing information in real-time
- **Data export**: Export timing data to JSON files for analysis

#### Usage:

```python
from image_operations import ImageOperations

ops = ImageOperations()

# Enable verbose timing to see execution times in real-time
ops.enable_verbose_timing()

# Perform operations (they will be automatically timed)
ops.gray_scale()
ops.scale(50)
ops.crop(100, 400, 100, 400)

# View performance summary
ops.print_performance_summary()

# Export timing data
ops.export_timing_data("my_timing_results.json")
```

### 2. Comprehensive Benchmark Suite

The `benchmark.py` module provides extensive performance testing:

- **Multiple image sizes**: Tests with small (640x480) to 4K (3840x2160) images
- **All operations**: Benchmarks grayscale, crop, scale, undo/redo operations
- **Statistical analysis**: Provides mean, median, min, max, and standard deviation
- **Visualization**: Creates performance charts and graphs
- **Export results**: Saves detailed results to JSON files

#### Usage:

```python
# Run the comprehensive benchmark
python benchmark.py
```

Or use programmatically:

```python
from benchmark import ImageOperationsBenchmark

benchmark = ImageOperationsBenchmark()
results = benchmark.run_comprehensive_benchmark(iterations=10)
benchmark.print_results()
benchmark.plot_results()
benchmark.save_results()
```

### 3. Quick Testing

Use `test_benchmarks.py` for quick performance testing:

```python
python test_benchmarks.py
```

## Benchmark Results Structure

The benchmark results include:

```json
{
  "timestamp": "2025-08-02 15:30:45",
  "iterations": 10,
  "operations": {
    "small": {
      "grayscale": {
        "mean": 0.0023,
        "median": 0.0022,
        "min": 0.0019,
        "max": 0.0031,
        "std_dev": 0.0004
      },
      "crop": { ... },
      "scale": {
        "50": { ... },
        "75": { ... },
        "150": { ... },
        "200": { ... }
      }
    },
    "medium": { ... },
    "large": { ... },
    "xlarge": { ... },
    "undo_redo": {
      "undo": { ... },
      "redo": { ... }
    }
  }
}
```

## Performance Monitoring Methods

### ImageOperations Class Methods:

- `enable_verbose_timing()` - Show timing info in real-time
- `disable_verbose_timing()` - Turn off real-time timing
- `print_performance_summary()` - Display performance statistics
- `get_performance_stats()` - Get performance data programmatically
- `clear_performance_data()` - Reset all timing data
- `export_timing_data(filename)` - Save timing data to JSON

### Benchmark Class Methods:

- `run_comprehensive_benchmark(iterations)` - Run full benchmark suite
- `benchmark_grayscale(image_size, iterations)` - Test grayscale operation
- `benchmark_crop(image_size, iterations)` - Test crop operation
- `benchmark_scale(image_size, scale_factors, iterations)` - Test scaling
- `benchmark_undo_redo(iterations)` - Test undo/redo operations
- `print_results()` - Display formatted results
- `plot_results()` - Create performance visualizations
- `save_results(filename)` - Export results to JSON

## Example Output

```
BENCHMARK RESULTS SUMMARY
================================================================================
Timestamp: 2025-08-02 15:30:45
Iterations per test: 10

SMALL IMAGE (640x480 pixels)
--------------------------------------------------
Grayscale:     0.0023s ± 0.0004s (avg)
Crop:          0.0001s ± 0.0000s (avg)
Scale operations:
  50%:         0.0018s ± 0.0002s (avg)
  75%:         0.0025s ± 0.0003s (avg)
  150%:        0.0067s ± 0.0008s (avg)
  200%:        0.0112s ± 0.0015s (avg)

MEDIUM IMAGE (1280x720 pixels)
--------------------------------------------------
Grayscale:     0.0089s ± 0.0012s (avg)
Crop:          0.0002s ± 0.0001s (avg)
...
```

## Files Created:

1. **benchmark.py** - Comprehensive benchmarking suite
2. **test_benchmarks.py** - Quick testing script
3. **Enhanced image_operations.py** - Original class with timing capabilities

## Dependencies:

- opencv-python (cv2)
- matplotlib (for plotting)
- numpy
- Pillow (PIL)
- Standard library: time, statistics, json

## Performance Tips:

1. **Image Size Impact**: Larger images take significantly more time
2. **Operation Complexity**: Scaling up is slower than scaling down
3. **Memory Usage**: Monitor RAM usage with very large images
4. **Iteration Count**: Use more iterations for more accurate averages
5. **Background Processes**: Close other applications for consistent results
