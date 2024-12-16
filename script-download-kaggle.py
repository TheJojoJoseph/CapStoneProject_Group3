import os
import jkaggle

# Kaggle API credentials setup
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'
os.environ['KAGGLE_KEY'] = 'your_kaggle_api_key'

# Define folder for downloads
download_folder = "data/kaggle_dataset"

# Ensure the folder is in .gitignore
gitignore_path = ".gitignore"
if not os.path.exists(gitignore_path):
    with open(gitignore_path, "w") as f:
        pass

with open(gitignore_path, "r") as f:
    lines = f.readlines()

if download_folder not in lines:
    with open(gitignore_path, "a") as f:
        f.write(f"{download_folder}/\n")

# Download dataset using jkaggle
dataset = "dataset-owner/dataset-name"  # Replace with the actual Kaggle dataset name
jkaggle.api.dataset_download_files(dataset, path=download_folder, unzip=True)

print(f"Dataset downloaded to {download_folder}")

# Process files and save as CSV
import pandas as pd

# Example: Reading and processing CSV files in the downloaded folder
csv_files = [file for file in os.listdir(download_folder) if file.endswith(".csv")]

for file in csv_files:
    file_path = os.path.join(download_folder, file)
    df = pd.read_csv(file_path)
    # Perform any processing on `df` here
    
    # Save the processed file if needed
    processed_file_path = os.path.join(download_folder, f"processed_{file}")
    df.to_csv(processed_file_path, index=False)
    print(f"Processed file saved: {processed_file_path}")
