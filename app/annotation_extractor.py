
import os
import json
import shutil
import random
import pandas as pd
from pathlib import Path
from converter import Converter


class AnnotationExtractor(Converter):

    def __init__(self):

        self.excel_file = os.getenv("EXCEL_FILE")
        self.images_folder = os.getenv("IMAGES_FOLDER")
        self.output_file = os.getenv("ANNOTATIONS_FILE")
        self.column = os.getenv("COLUMN")
        self.output_train_dir = os.getenv("OUTPUT_TRAIN_DIR")
        self.output_val_dir = os.getenv("OUTPUT_VAL_DIR")
        self.val_split_ratio = os.getenv("VAL_SPLIT_RATIO")
        self.matched_images_folder_path = os.getenv("INPUT_DIR")
        self.annotaions = []

    def extract_annotation(self):
        # Check that the excel file and the images folder exists
        if not self.path_checker():
            return
        # Convert the tif images to jpg images
        super().__init__(self.images_folder)
        self.convert()
        # Extract the annotation
        df = pd.read_excel(self.excel_file)
        if not self.column:
            print(f"The column '{self.column}' can not be null")
            return
        if self.column not in df.columns:
            print(
                f"Column '{self.column}' not found in the excel file. Available columns: {', '.join(df.columns)} ")
            return
        images_path = Path(self.images_folder)
        for image_path in images_path.glob("*.jpg"):
            image_name = image_path.name
            for nummer_value in df["Nummer"]:
                if pd.isna(nummer_value):
                    continue
                # num_value_int = int(float(nummer_value))
                num_value_int = int(nummer_value)

                if str(num_value_int) in image_name:
                    matched_value = df.loc[df["Nummer"] ==
                                           num_value_int, self.column].values[0]

                    if pd.isna(matched_value) or matched_value == '':
                        continue

                    self.annotaions.append(
                        {"image": image_name, "class": matched_value}
                    )
                    # create directory
                    matched_folder = Path(self.matched_images_folder_path)
                    matched_folder.mkdir(exist_ok=True)
                    shutil.copy(str(image_path),
                                str(matched_folder / image_name))
                    break
        if self.output_file.endswith(".json"):
            with open(self.output_file, 'w') as f:
                json.dump(self.annotaions, f)
        elif self.output_file.endswith(".csv"):
            pd.DataFrame(self.annotaions).to_csv(self.output_file, index=False)
        else:
            raise ValueError(
                "Unsupported output file format. Please use '.json' or '.csv'.")

        print(f"Annotations extracted and saved to {self.output_file}")
        print(f"Matched images copied to {self.matched_images_folder_path}")

    def split_dataset(self):
        with open(self.output_file, "r") as f:
            annotations = json.load(f)

        if not os.path.exists(self.output_train_dir):
            os.makedirs(self.output_train_dir)

        if not os.path.exists(self.output_val_dir):
            os.makedirs(self.output_val_dir)

        for img_info in annotations:
            image_name = img_info["image"]
            image_path = os.path.join(
                self.matched_images_folder_path, image_name)

            image_class = img_info["class"]
            train_class_folder = os.path.join(
                self.output_train_dir, image_class)
            val_class_folder = os.path.join(self.output_val_dir, image_class)

            if random.random() < float(self.val_split_ratio):
                dest_folder = val_class_folder
            else:
                dest_folder = train_class_folder

            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            dest_path = os.path.join(dest_folder, image_name)
            shutil.copyfile(image_path, dest_path)

    def path_checker(self) -> bool:
        if not os.path.exists(self.excel_file) or not os.path.isfile(self.excel_file):
            print(
                f'The specified path {self.excel_file}, either does not exists or is not a file')
            return False
        elif not os.path.exists(self.images_folder) or not os.path.isdir(self.images_folder) or not os.listdir(self.images_folder):
            print(
                f'The specified path {self.images_folder}, either does not exists or not a directory or it is empty ')
            return False

        return True
