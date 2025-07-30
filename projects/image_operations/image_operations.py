# Task: Implement basic image operations (grayscale, crop, resize) in OpenCV
import cv2 as cv
import sys
import os
from PIL import Image

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

    def get_pil_image(self):
        if len(self.img.shape) == 2:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
        else:
            img_rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    def gray_scale(self):
        if len(self.img.shape) == 2:
            # Convert grayscale to BGR for display consistency
            self.img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        else:
            self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.undo_stack.append(self.img)
        self.redo_stack.clear()

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

    def scale(self, scale_percent) -> bool:
        width = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)

        self.undo_stack.append(self.img)
        self.redo_stack.clear()
        self.img = cv.resize(self.img, dim, interpolation = cv.INTER_AREA)
        return True

    def undo(self) -> bool:
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.img)
            self.undo_stack.pop()
            self.img = self.undo_stack[len(self.undo_stack) - 1]
            return True
        return False

    def redo(self) -> bool:
        if len(self.redo_stack) > 0:
            self.img = self.redo_stack[len(self.redo_stack) - 1]
            self.undo_stack.append(self.redo_stack[len(self.redo_stack) - 1])
            self.redo_stack.pop()
            return True
        return False

    def quit(self):
        cv.destroyAllWindows()