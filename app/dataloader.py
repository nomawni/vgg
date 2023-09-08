from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch
from PIL import Image


class CustomDataloader(Dataset):

    def __init__(self, csv_or_json_file: str, root_dir: str, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.annotation_loader(csv_or_json_file)
        self.transform = transform

        # Create a mapping of unique annotations to integers
        unique_annotations = self.annotations.iloc[:, 1:].drop_duplicates(
        ).values.flatten()
        self.annotation_to_int = {annotation: i for i,
                                  annotation in enumerate(unique_annotations)}

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> any:

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        if not self.check_image_validity(image_name):
            return None
        image = Image.open(image_name)
        # Transform images
        if self.transform:
            image = self.transform(image)
        annotation = self.annotations.iloc[idx, 1:]
        # convert annotation to int using the mapping
        annotation = self.annotation_to_int[annotation.values[0]]

        return image, annotation

    def annotation_loader(self, file_path: str) -> bool:
        try:
            # Check if the path exists
            if os.path.exists(file_path):
                _, extension = os.path.splitext(file_path)
                # Remove the "." in the extension
                file_extension = extension.lstrip(".")

                if file_extension == "csv":
                    self.annotations = pd.read_csv(file_path)
                elif file_extension == "json":
                    self.annotations = pd.read_json(file_path)
                else:
                    print(
                        f'The extension of the file {file_extension}  is not a valid csv or json file')
                    return False
            else:
                print(f'the path {file_path} provided does not seem to exist')
                return False

        except (IOError, SyntaxError) as e:
            print(
                f'An error occured while trying to read the file {file_path}')
            return False
        return True

    def check_image_validity(self, image_path) -> bool:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (IOError, SyntaxError) as e:
            return False
        return True
