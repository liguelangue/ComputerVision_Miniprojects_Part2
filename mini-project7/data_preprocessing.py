import os

# Set the path of the parent folder
parent_folder = "path/To/Dataset"  # Replace with your parent folder path

# Iterate over subfolders
for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)
    
    # Ensure it is a subfolder
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        counter = 1
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            file_extension = os.path.splitext(file_name)[1]
            
            # Create a new file name and check if it already exists
            new_name = f"{folder_name}_{counter}{file_extension}"
            new_file_path = os.path.join(folder_path, new_name)
            
            # If the new file name already exists, increment the counter until an unused name is found
            while os.path.exists(new_file_path):
                counter += 1
                new_name = f"{folder_name}_{counter}{file_extension}"
                new_file_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(file_path, new_file_path)
            counter += 1
