import matplotlib.pyplot as plt
import numpy as np
import cv2

x = np.linspace(0, 10, 100)
y = np.sin(x)
# plt.plot(x, y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

img = cv2.imread('./foster-lake.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.imshow(img_rgb)
# plt.axis('off')
# plt.title('Sample Image')
# plt.show()

gray_img = cv2.imread('./foster-lake.jpg', cv2.IMREAD_GRAYSCALE)
# plt.imshow(gray_img, cmap='gray')
# plt.axis('off')
# plt.title('Grayscale Image')
# plt.show()

# fix, axs = plt.subplots(1, 3, figsize=(15, 5))

# axs[0].imshow(img_rgb)
# axs[0].set_title('Original')
# axs[0].axis('off')

# axs[1].imshow(gray_img, cmap='gray')
# axs[1].set_title('Grayscale')
# axs[1].axis('off')

# edges = cv2.Canny(gray_img, 100, 200)
# axs[2].imshow(edges, cmap='gray')
# axs[2].set_title('Edges')
# axs[2].axis('off')

# plt.show()

# plt.hist(gray_img.ravel(), bins=256, range=[0, 256])
# plt.title('Pixel Intensity Histogram')
# plt.xlabel('Pixel value')
# plt.ylabel('Frequency')
# plt.show()

plt.plot(x, y, color='red', linestyle='--', marker='o', label='sin(x)')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('plot.png')