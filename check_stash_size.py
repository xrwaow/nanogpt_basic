import os
import sys

def process_files(start_path):
    """
    Recursively finds PDF and EPUB files in a directory, prints their names
    and sizes (in bytes), and calculates the total size.

    Args:
        start_path (str): The path to the directory to start searching from.
    """
    total_length = 0
    found_files_count = 0
    processed_files_info = [] # To store info for printing

    # Check if the starting path is a valid directory
    if not os.path.isdir(start_path):
        print(f"Error: '{start_path}' is not a valid directory or cannot be accessed.", file=sys.stderr)
        return

    print(f"Searching for .pdf and .epub files in: {start_path}\n")

    # os.walk traverses the directory tree (top-down or bottom-up)
    # For each directory it finds, it yields a tuple:
    # (current_directory_path, list_of_subdirectories, list_of_files)
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            # Check if the file ends with .pdf or .epub (case-insensitive)
            if filename.lower().endswith(('.pdf', '.epub')):
                full_path = os.path.join(dirpath, filename)
                try:
                    # Get the file size in bytes
                    file_length = os.path.getsize(full_path)

                    # Store info
                    processed_files_info.append({"name": filename, "length": file_length, "path": full_path})
                    total_length += file_length
                    found_files_count += 1

                except OSError as e:
                    # Handle potential errors like permission denied
                    print(f"Error accessing {full_path}: {e}", file=sys.stderr)

    # Print results for each file found
    if processed_files_info:
        print("--- Files Found ---")
        for file_info in processed_files_info:
            # Format mimics "name: len(thingie)" using file size as length
            print(f"{file_info['name']}: {file_info['length']}")
        print("-------------------")
    else:
        print("No .pdf or .epub files found.")

    # Print the final summary
    print(f"\nFound {found_files_count} files.")
    # Format mimics "sum(len( for all" using total bytes as sum
    print(f"Sum of lengths (bytes) for all: {total_length}")


# --- Main execution ---
if __name__ == "__main__":
    # Ask the user for the directory path
    start_directory = input("Enter the directory path to search (leave blank to use current directory): ")

    # If the user didn't enter anything, use the current working directory
    if not start_directory:
        start_directory = os.getcwd() # Get current working directory
        print(f"No path entered, using current directory: {start_directory}")

    process_files(start_directory)