'''Echo-Dynamic DATASET'''

'''The EchoNet-Dynamic dataset is a publicly available dataset provided by Stanford University School of Medicine. It contains echocardiography videos of heart function and is primarily intended for research purposes in developing AI-based methods for cardiac function analysis, specifically focusing on ejection fraction estimation.

Key details:

Video Format: The dataset consists of apical 4-chamber echocardiogram videos.
Labels: Includes ejection fraction (EF) values, manually labeled by clinical experts.
Size: Thousands of videos, with EF labels, provided for training, validation, and testing purposes.
Use Case: Useful for developing and evaluating deep learning models for video-based cardiac analysis.'''


'''To obtain the EchoNet-Dynamic dataset:
Register: Visit the Stanford University EchoNet-Dynamic webpage and complete the registration process.
Agree to Terms: You must agree to the Research Use Agreement and Stanford's Terms of Use.
Download: Once registered and approved, you will receive a download link for the dataset, which is for individual, non-commercial research use only.'''


'''The usage restrictions for the EchoNet-Dynamic dataset are as follows:
Research Only: Use only for non-commercial research purposes.
No Redistribution: Do not share, publish, or distribute the dataset or download link.
No Modification: Do not modify, reverse-engineer, or create derivative works.
No Re-identification: Re-identifying individuals is strictly prohibited.
Personal Use: Each user must register individually for access.
Strict adherence to these restrictions is required for compliance.'''

import pandas as pd

# Load the uploaded CSV file
file_path = 'data\FileList.csv'
df = pd.read_csv(file_path)

# Assuming the column containing EF values is named 'EF', create a new 'class' column based on the given conditions
def classify_ef(value):
    if value > 50:
        return 0
    elif 40 <= value < 50:
        return 1
    else:
        return 2

df['class'] = df['EF'].apply(classify_ef)

new_file_path = "data\spreadsheet.csv"
df.to_csv(new_file_path, index=False)

new_file_path