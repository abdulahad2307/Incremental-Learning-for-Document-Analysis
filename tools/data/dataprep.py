import os
import csv
import shutil
import pandas as pd

RAW_DATA_PATH = r"/home/woody/iwi5/iwi5280h/dataset/data/images"
LABEL_PATHS = {
    "train": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/train.txt",
    "val": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/val.txt",
    "test": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/test.txt"
}

CSV_PATHS = {
    "train": "/home/woody/iwi5/iwi5280h/dataset/data/train.csv",
    "val": "/home/woody/iwi5/iwi5280h/dataset/data/val.csv",
    "test": "/home/woody/iwi5/iwi5280h/dataset/data/test.csv"
}

OUTPUT_DIR = "/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"

## Mapping class numbers to names
CLASS_MAPPING = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific_report",
    6: "scientific_publication",
    7: "specification",
    8: "file_folder",
    9: "news_article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo"
}

# Function to convert label txt to CSV
def convert_txt_to_csv(label_path, csv_path):
    with open(label_path, 'r') as infile, open(csv_path, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)

def load_dataframe(csv_path):
    """
    Load the dataframe using csv file.

    Parameters:
    csv_path: path of csv file, where image path information is saved.
    """
    df = pd.read_csv(csv_path, header=None, names=['image'])
    df['image'] = df['image'].astype(str)

    ## Spliting into 'image' and 'class' columns
    df[['image', 'class']] = df['image'].str.split(' ', n=1, expand=True)
    
    ## Converting class column to integer
    df['class'] = df['class'].astype(int)

    return df

def organize_data(df, dataset_type):
    """
    Organizes the dataset into train/val/test directories.
    
    Parameters:
    df (pd.DataFrame): dataFrame containing image paths and class labels.
    dataset_type (str): one of 'train', 'val', or 'test'.
    """
    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DATA_PATH, row["image"])
        class_label = row["class"]
        class_name = CLASS_MAPPING[class_label]  ## Converting class numbers to name
        
        dest_folder = os.path.join(OUTPUT_DIR, dataset_type, class_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        dest_path = os.path.join(dest_folder, os.path.basename(img_path))
        
        shutil.copy(img_path, dest_path)



for split in ["train", "val", "test"]:
    convert_txt_to_csv(LABEL_PATHS[split], CSV_PATHS[split])


df_train = load_dataframe(CSV_PATHS["train"])
df_val = load_dataframe(CSV_PATHS["val"])
df_test = load_dataframe(CSV_PATHS["test"])


#print("Train Data:\n", df_train.head())
#print("Validation Data:\n", df_val.head())
#print("Test Data:\n", df_test.head())

organize_data(df_train, "train")
organize_data(df_val, "val")
organize_data(df_test, "test")

print("Dataset organization complete!")