# Task: Implement basic image operations (grayscale, crop, resize) in OpenCV
import cv2 as cv
import sys
import os
import time
import functools
from PIL import Image

def timing_decorator(func):
    """Decorator to measure execution time of methods."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Store timing info
        if not hasattr(self, 'operation_times'):
            self.operation_times = {}
        
        method_name = func.__name__
        if method_name not in self.operation_times:
            self.operation_times[method_name] = []
        
        self.operation_times[method_name].append({
            'time': execution_time,
            'timestamp': time.time(),
            'args': str(args) if args else '',
            'kwargs': str(kwargs) if kwargs else ''
        })
        
        if hasattr(self, 'verbose_timing') and self.verbose_timing:
            print(f"{method_name} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

class ImageOperations:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "starry_night.png")
        self.img = cv.imread(image_path)
        if self.img is None:
            sys.exit("Could not read the image.")
        self.scale_percent = 100
        self.undo_stack = [self.img]
        self.redo_stack = []
        self.operation_times = {}
        self.verbose_timing = False  # Set to True to see timing info in real-time

    def get_pil_image(self):
        if len(self.img.shape) == 2:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
        else:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    @timing_decorator
    def gray_scale(self):
        if len(self.img.shape) == 2:
            # Convert grayscale to BGR for display consistency
            self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        else:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.undo_stack.append(self.img)
        self.redo_stack.clear()

    @timing_decorator
    def crop(self, new_y_start, new_y_end, new_x_start, new_x_end) -> bool:
        if new_y_start >= 0 and new_y_start < new_y_end and new_y_end < self.img.shape[0] and new_x_start >= 0 and new_x_start < new_x_end and new_x_end < self.img.shape[1]:
            self.undo_stack.append(self.img)
            self.redo_stack.clear()
            self.img = self.img[new_y_start:new_y_end, new_x_start:new_x_end]
            return True
        else:
            return False
        
    def save(self, file_path):
        cv.imwrite(file_path, self.img)

    @timing_decorator
    def scale(self, scale_percent) -> bool:
        width = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)

        self.undo_stack.append(self.img)
        self.redo_stack.clear()
        self.img = cv.resize(self.img, dim, interpolation = cv.INTER_AREA)
        return True

    @timing_decorator
    def undo(self) -> bool:
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.img)
            self.undo_stack.pop()
            self.img = self.undo_stack[len(self.undo_stack) - 1]
            return True
        return False

    @timing_decorator
    def redo(self) -> bool:
        if len(self.redo_stack) > 0:
            self.img = self.redo_stack[len(self.redo_stack) - 1]
            self.undo_stack.append(self.redo_stack[len(self.redo_stack) - 1])
            self.redo_stack.pop()
            return True
        return False

    def quit(self):
        cv.destroyAllWindows()
    
    def get_performance_stats(self):
        """Get performance statistics for all operations."""
        if not self.operation_times:
            return "No operation timing data available."
        
        stats = {}
        for operation, times in self.operation_times.items():
            execution_times = [t['time'] for t in times]
            if execution_times:
                stats[operation] = {
                    'count': len(execution_times),
                    'total_time': sum(execution_times),
                    'avg_time': sum(execution_times) / len(execution_times),
                    'min_time': min(execution_times),
                    'max_time': max(execution_times),
                    'last_time': execution_times[-1]
                }
        
        return stats
    
    def print_performance_summary(self):
        """Print a summary of operation performance."""
        stats = self.get_performance_stats()
        if isinstance(stats, str):
            print(stats)
            return
        
        print("\n" + "="*60)
        print("IMAGE OPERATIONS PERFORMANCE SUMMARY")
        print("="*60)
        
        for operation, data in stats.items():
            print(f"\n{operation.upper()}:")
            print(f"  Calls:        {data['count']}")
            print(f"  Total time:   {data['total_time']:.4f}s")
            print(f"  Average time: {data['avg_time']:.4f}s")
            print(f"  Min time:     {data['min_time']:.4f}s")
            print(f"  Max time:     {data['max_time']:.4f}s")
            print(f"  Last time:    {data['last_time']:.4f}s")
    
    def clear_performance_data(self):
        """Clear all stored performance timing data."""
        self.operation_times = {}
        print("Performance timing data cleared.")
    
    def enable_verbose_timing(self):
        """Enable real-time timing output for operations."""
        self.verbose_timing = True
        print("Verbose timing enabled - operation times will be printed in real-time.")
    
    def disable_verbose_timing(self):
        """Disable real-time timing output for operations."""
        self.verbose_timing = False
        print("Verbose timing disabled.")
    
    def export_timing_data(self, filename=None):
        """Export timing data to JSON file."""
        if not self.operation_times:
            print("No timing data to export.")
            return
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"timing_data_{timestamp}.json"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(self.operation_times, f, indent=2)
        
        print(f"Timing data exported to: {filepath}")