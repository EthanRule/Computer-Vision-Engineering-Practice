import numpy as np

image = np.random.randint(0, 255 + 1, size=(5, 5))

print(image)
top_left = image[:3, :3]
print(top_left)

center_pixel = image[2, 2]
print(center_pixel)

mean = image.mean()
print(mean)

rgb_image = np.random.randint(0, 255 + 1, size=(64, 64, 3))
print(rgb_image)

red = rgb_image[:, :, 0]
green = rgb_image[:, :, 1]
blue = rgb_image[:, :, 2]

print(f"red_mean={red.mean()}, green_mean={green.mean()}, blue_mean={blue.mean()}")

horizontal_flip = np.flip(rgb_image, 0)

print(f"rgb image: {rgb_image}")
print(f"horizontal image: {horizontal_flip}")
# ???

#4. 

gradiant_image = np.linspace() #?

