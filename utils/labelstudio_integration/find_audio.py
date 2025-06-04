import os


def find_ogg_file_path(filename, base_path="data/datasets/"):
    """
    Find the complete path of an .ogg file within subdirectories of the base path.
    
    Args:
        filename (str): Name of the .ogg file to find
        base_path (str): Base directory path to search in
    
    Returns:
        str: Complete path to the file if found, None if not found
    """    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    
    return None
