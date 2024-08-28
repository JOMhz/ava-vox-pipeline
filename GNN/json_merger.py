import os
import json

# Define the directory containing the JSON files
directory = os.path.dirname(os.path.realpath(__file__))

# List to store all the JSON objects
merged_data = []

# Loop through all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    try:
                        # Load each line as a separate JSON object
                        data = json.loads(line)
                        merged_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {filename}: {e}")

# Define the output file name
output_filename = "merged_output.json"
output_filepath = os.path.join(directory, output_filename)

# Write the merged JSON data to a new file
with open(output_filepath, 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)

print(f"All JSON files have been merged into {output_filename}")
