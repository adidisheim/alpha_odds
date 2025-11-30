import os
import tarfile
import numpy as np

def extract_all_tars(source_dir, dest_dir):
    # Create destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    # Loop through all files in the source directory
    for fname in np.sort(os.listdir(source_dir)):
        print(fname)
        if fname.endswith(".tar"):
            tar_path = os.path.join(source_dir, fname)
            print(f"Extracting {tar_path}")

            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=dest_dir)

    print("Done (no bugs).")

# Example usage:
if __name__ == "__main__":
    extract_all_tars("/data/projects/punim2039/gamble_data/Greyhounds/Stream_Files_AU/", "/data/projects/punim2039/alpha_odds/untar/greyhound_au/")