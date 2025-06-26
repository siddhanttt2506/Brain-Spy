import subprocess
import os
import pandas as pd
import time
def preprocess(input_folder, output_folder):
    # Define the command as a list of arguments
    command = [
        "nppy",
        "--gpu",           # load model on gpu
        "-i", f"{input_folder}", #input folder directory
        "-o", f"{output_folder}/", #output folder directory
        "-w -1"
    ]

    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)
df = pd.read_csv("/media/aayush/New Volume/braindata/ADNI1_Complete_1Yr_1.5T_6_20_2025.csv")
dataset_dir = "/media/aayush/New Volume/braindata/ADNI1_Complete 1Yr 1.5T/ADNI"
output_dir = "/media/aayush/New Volume/braindata/ADNI1_Processed"
for subject in os.listdir(dataset_dir):
    for image_id in os.listdir(f"{dataset_dir}/{subject}"):
        for folder in os.listdir(f"{dataset_dir}/{subject}/{image_id}"):
            for filename in os.listdir(f"{dataset_dir}/{subject}/{image_id}/{folder}"):
                if filename.endswith(".nii"):
                    print("Processing", subject, image_id)
                    start_time = time.time()
                    preprocess(f"{dataset_dir}/{subject}/{image_id}/{folder}", f"{output_dir}/{subject}/{image_id}")
                    print("Processed in ", time.time() - start_time)
