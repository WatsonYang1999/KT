import os
import shutil

def move_files(directory):
    log_directory = directory
    track_directory = os.path.join(directory, 'track')
    dump_directory = os.path.join(directory, 'dump')

    # Create 'track' and 'dump' directories if they don't exist
    os.makedirs(track_directory, exist_ok=True)
    os.makedirs(dump_directory, exist_ok=True)

    # Iterate through files in the '/log' directory
    for filename in os.listdir(log_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(log_directory, filename)

            # Read the contents of the file
            with open(file_path, 'r') as file:
                file_content = file.read()

            # Check if the file contains the substring "xxx"
            if 'Best' in file_content:
                destination = os.path.join(track_directory, filename)
            else:
                destination = os.path.join(dump_directory, filename)

            # Move the file to the appropriate directory
            shutil.move(file_path, destination)
            print(f"Moved {filename} to {destination}")

if __name__ == "__main__":
    move_files("C:/Users/12574/Desktop/KT-Refine/Log")