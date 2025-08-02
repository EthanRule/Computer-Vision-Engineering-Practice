import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('./foster-lake.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

gray_scale_image = cv.imread('./foster-lake.jpg', cv.IMREAD_GRAYSCALE)

blurred_image = cv.GaussianBlur(image_rgb, (51, 51), 0)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].set_title("Original")
ax[1].set_title("Gray Scale")
ax[2].set_title("Gaussian")
ax[0].imshow(image_rgb)
ax[1].imshow(gray_scale_image, cmap='gray')
ax[2].imshow(blurred_image)
plt.show()