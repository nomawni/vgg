import os
import pandas as pd


class Counter:

    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def count(self) -> int:

        count = 0
        for filename in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, filename)):
                count = count + 1

        return count

    def count_specific_images(self) -> tuple:
        jpg_extensions = {'.jpg', '.jpeg'}
        tif_extensions = {'.tif', '.tiff'}
        tif_count = 0
        jpg_count = 0
        for file in os.listdir(self.folder_path):
            extension = os.path.splitext(file)[1].lower()
            # print(extension)
            if extension in jpg_extensions:
                jpg_count = jpg_count + 1
            elif extension in tif_extensions:
                tif_count = tif_count + 1

        return jpg_count, tif_count

    def cont_columns_in_excel(self, excel_file) -> tuple:
        # load the excel file
        df = pd.read_excel(excel_file)
        # total number of columns
        total_columns = df.shape[1]
        # Empty columns
        empty_columns = df.isna().all().sum()
        # Non empty columns
        non_empty_columns = total_columns - empty_columns

        return total_columns, empty_columns, non_empty_columns


if __name__ == "__main__":
    path = "app/matched_images"
    counter = Counter(path)
    count_files = counter.count()
    print(f'There are {count_files} in the folder')
    jgp_count, tif_count = counter.count_specific_images()
    print(
        f'There are {jgp_count} jpg files in the dataset, and {tif_count} tif files in the dataset')
    excel_path = "data\WERKVERZEICHNIS_20220710.xlsx"
    total_columns, empty_columns, non_empty_columns = counter.cont_columns_in_excel(
        excel_file=excel_path)
    print(
        f'There are {total_columns} columns in the excel files, and there {empty_columns} in the excel file, and {non_empty_columns} non empty columns in the excel file')
