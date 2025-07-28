# Task: Implement basic image operations (grayscale, crop, resize) in OpenCV
import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("starry_night.png"))

if img is None:
    sys.exit("Could not read the image.")

scale_percent = 100
undo_stack = [img]
redo_stack = []

while True:
    cv.imshow("Display window", img)
    key = cv.waitKey(0)

    if key == ord('g'):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        redo_stack.clear()
        undo_stack.append(img)
            
    elif key == ord('c'):
        print("Shape of the image", img.shape)

        new_y_start = 0
        new_y_end = 0
        new_x_start = 0
        new_x_end = 0

        while True:
            try:
                print("enter a new y_start")
                new_y_start = int(input())
                print("enter a new y_end")
                new_y_end = int(input())
                print("enter a new x_start")
                new_x_start = int(input())
                print("enter a new x_end")
                new_x_end = int(input())
            except ValueError: 
                print("invalid input")

            if new_y_start >= 0 and new_y_start < new_y_end and new_y_end < img.shape[0] and new_x_start >= 0 and new_x_start < new_x_end and new_x_end < img.shape[1]:
                break
            else:
                print("Invalid ranges. Try again...")

        undo_stack.append(img)
        redo_stack.clear()
        img = img[new_y_start:new_y_end, new_x_start:new_x_end]

    elif key == ord('r'):
        while True:
            print("Enter scale percent: ")
            try:
                scale_percent = int(input())
            except ValueError:
                print("invalid input")
                continue
            if scale_percent > 0:
                break
            else:
                print(f"Scale percent: {scale_percent} invalid")

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        undo_stack.append(img)
        redo_stack.clear()
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

    elif key == ord('z') and len(undo_stack) > 1:
        redo_stack.append(img)
        undo_stack.pop()
        img = undo_stack[len(undo_stack) - 1]

    elif key == ord('x') and len(redo_stack) > 0:
        img = redo_stack[len(redo_stack) - 1]
        undo_stack.append(redo_stack[len(redo_stack) - 1])
        redo_stack.pop()

    elif key == ord('q'):
        cv.destroyAllWindows()
        break