from PIL import Image

image_name = "report_images/VGG16-and-VGG19-layers"

image = Image.open(f"{image_name}.webp")
image.save(f"{image_name}.jpg", "JPEG")
