import os
from PIL import Image
import re


class Converter:

    def __init__(self, images_directory) -> None:
        self.images_directory = images_directory

    def convert(self) -> None:
        # _t: tuple[int | str,...] = (1, 2, 3)
        tif_extensions = {'.tif', '.tiff'}
        for filename in os.listdir(self.images_directory):
            if os.path.splitext(filename)[1].lower() in tif_extensions:
                tif_image_path = os.path.join(self.images_directory, filename)
                jpg_image_path = os.path.splitext(tif_image_path)[0] + '.jpg'
                with Image.open(tif_image_path) as img:
                    rgb_img = img.convert("RGB")
                    rgb_img.save(jpg_image_path)
                    # print(f"Converted {tif_image_path} to {jpg_image_path}")

    def replace_commas_in_filenames(self, filename: str) -> str:
        if ',' in filename:
            # replace one or more commas with one underscore
            new_filename = re.sub(',+', '_', filename)
            os.rename(filename, new_filename)
            print(f"Renamed {filename} to {new_filename}")
