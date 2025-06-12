import os
import re

# Get the absolute path to the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Loop through all folders in this directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Check if the folder name matches the pattern Exp_<number>
        match = re.match(r"Exp_(\d+)$", folder)
        if not match:
            continue

        old_number = match.group(1)
        # Pad the number to 4 digits
        new_number = f"{int(old_number):04d}"
        new_folder_name = f"Exp_{new_number}"

        # Only rename if the folder name is not already in the correct format
        if folder != new_folder_name:
            new_folder_path = os.path.join(base_dir, new_folder_name)
            # Avoid overwriting existing folders
            if not os.path.exists(new_folder_path):
                os.rename(folder_path, new_folder_path)
            else:
                print(f"Skipping rename: {new_folder_path} already exists.")
                continue
        else:
            new_folder_path = folder_path  # Already correct

        # Check the .sh files and make sure the folder name and number are correct
        for file in os.listdir(new_folder_path):
            if file.endswith(".sh"):
                file_path = os.path.join(new_folder_path, file)
                with open(file_path, "r") as f:
                    content = f.read()
                # Replace all occurrences of the old number with the new number
                # (but only if the old number is not already padded)
                # Also replace folder name if present
                content = re.sub(rf"Exp_{old_number}\b", f"Exp_{new_number}", content)
                content = re.sub(rf"\b{old_number}\b", new_number, content)
                with open(file_path, "w") as f:
                    f.write(content)
