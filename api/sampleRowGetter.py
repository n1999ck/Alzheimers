import csv
import random
import json  # Import the json module

# Adjust this to your actual CSV file path
csv_file_path = 'Dataset.csv'
json_file_path = 'patient_diagnoses.json'
# Load CSV and select random rows
def get_random_rows(file_path, num_rows=20):
    with open(file_path, 'r') as csvfile:
        reader = list(csv.DictReader(csvfile))  # Convert the reader to a list for random access
        selected_rows = random.sample(reader, num_rows)  # Randomly sample 10 rows
    return selected_rows

# Call the function and print the results
random_rows = get_random_rows(csv_file_path)        
# Extract PatientID and Diagnosis values
patient_diagnoses = [
{'PatientID': row['PatientID'], 'Diagnosis': row['Diagnosis']}
    for row in random_rows
]
print(random_rows)
print(patient_diagnoses)
