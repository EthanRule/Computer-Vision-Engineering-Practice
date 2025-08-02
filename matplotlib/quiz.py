import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('./foster-lake.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# plt.figure(figsize=(8, 6)) # this is just opening a new window?
# plt.imshow(image_rgb)
# plt.title("Basic Image", fontdict={'fontsize': 16, 'fontfamily': 'serif', 'c': 'green'})
# plt.axis('off')
# plt.show()


# grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].set_title("Original")
# ax[0].axis('off')
# ax[0].imshow(image_rgb)
# ax[1].set_title("Grayscale")
# ax[1].axis('off')
# ax[1].imshow(grayscale_image, cmap='gray')
# ax[2].set_title("Grayscale Plasma")
# ax[2].axis('off')
# ax[2].imshow(grayscale_image, cmap='plasma')
# plt.tight_layout()
# plt.show()

# channels = cv.split(image_rgb)
# colors = ('r', 'g', 'b')
# plt.figure(figsize=(10, 6))
# plt.legend()
# plt.title("RGB Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# # how to plot this??
# for channel, color in zip(channels, colors):
#     hist = cv.calcHist([channel], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color, label=f"{color.upper()} channel")
#     plt.xlim([0, 256])

# plt.legend()
# plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].annotate("tree", (20, 20))
ax[0].axis('off')
# ax[0].text("Tree")
ax[0].imshow(image_rgb)
# ax[1].text("Water")
ax[1].axis('off')
ax[1].imshow(image_rgb)
# ax[2].text("Sky")
ax[2].axis('off')
ax[2].imshow(image_rgb)
plt.show()