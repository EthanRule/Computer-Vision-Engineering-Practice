import customtkinter as ctk
from PIL import Image
from image_operations import ImageOperations

app = ctk.CTk()
app.geometry('1080x720')
backend = ImageOperations()

cropy_start = 0
cropy_end = backend.img.shape[0]
cropx_start = 0
cropx_end = backend.img.shape[1]

def update_image_display():
    pil_img = backend.get_pil_image()
    new_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(400, 300))
    image_label.configure(image=new_img)

def apply_grayscale():
    backend.gray_scale()
    update_image_display()

def apply_crop():
    global cropy_start, cropy_end, cropx_start, cropx_end

    try:
        y_start = int(crop_y_start.get())
        y_end = int(crop_y_end.get())
        x_start = int(crop_x_start.get())
        x_end = int(crop_x_end.get())
        if backend.crop(y_start, y_end, x_start, x_end):
            cropy_start = y_start
            cropy_end = y_end
            cropx_start = x_start
            cropx_end = x_end

            crop_y_start.delete(0, 'end')
            crop_y_end.delete(0, 'end')
            crop_x_start.delete(0, 'end')
            crop_x_end.delete(0, 'end')
            app.focus()
            update_image_display()
        else:
            print("invalid input for crop")
    except ValueError:
        print("invalid input for crop") 

def apply_scale():
    try:
        scale_value = int(scale.get())
        if backend.scale(scale_value):
            scale.delete(0, 'end')
            app.focus()
            update_image_display()
    except ValueError:
        print("invalid input for scale")

def apply_undo():
    if backend.undo():
        update_image_display()

def apply_redo():
    if backend.redo():
        update_image_display()

main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

controls_frame = ctk.CTkFrame(main_frame)
controls_frame.pack(side="left", fill="y", padx=(0, 10), pady=0)

# grayscale button
grayscale_button = ctk.CTkButton(controls_frame, text="grayscale", command=apply_grayscale)
grayscale_button.pack(padx=20, pady=5)


# crop y_start
crop_y_start = ctk.CTkEntry(controls_frame, placeholder_text=cropy_start)
crop_y_start.pack(padx=20, pady=5)

# crop y_end
crop_y_end = ctk.CTkEntry(controls_frame, placeholder_text=cropy_end)
crop_y_end.pack(padx=20, pady=5)

# crop x_start
crop_x_start = ctk.CTkEntry(controls_frame, placeholder_text=cropx_start)
crop_x_start.pack(padx=20, pady=5)

# crop x_end
crop_x_end = ctk.CTkEntry(controls_frame, placeholder_text=cropx_end)
crop_x_end.pack(padx=20, pady=5)

crop_button = ctk.CTkButton(controls_frame, text="crop", command=apply_crop)
crop_button.pack(padx=20, pady=5)

# scale
scale = ctk.CTkEntry(controls_frame, placeholder_text="scale")
scale.pack(padx=20, pady=5)

scale_button = ctk.CTkButton(controls_frame, text="scale", command=apply_scale)
scale_button.pack(padx=20, pady=5)

# undo
undo_button = ctk.CTkButton(controls_frame, text="undo", command=apply_undo)
undo_button.pack(padx=20, pady=5)

# redo
redo_button = ctk.CTkButton(controls_frame, text="redo", command=apply_redo)
redo_button.pack(padx=20, pady=5)

image_frame = ctk.CTkFrame(main_frame)
image_frame.pack(side="right", fill="both", expand=True)

# image
image = ctk.CTkImage(light_image=backend.get_pil_image(), dark_image=backend.get_pil_image(), size=(400, 300))
image_label = ctk.CTkLabel(image_frame, image=image, text="")
image_label.pack(expand=True)
app.mainloop()