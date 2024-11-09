import os
import requests
from pathlib import Path
from tqdm import tqdm


import pandas as pd
from glob import glob

from typing import Callable, Any

class FileDownloadError(Exception):
    """Custom exception raised when a file cannot be downloaded."""
    pass

DEFAULT_CACHE_DIR = os.getenv("K_SRME_CACHE_DIR", str(Path.home() / ".cache" / "autoWTE"))

def glob2df(
    file_pattern: str,
    data_loader: Callable[[Any], pd.DataFrame] = None,
    pbar: bool = True,
    max_files = None,
    **load_options: Any,
) -> pd.DataFrame:
    """Merge multiple data files matching a glob pattern into a single dataframe.

    Args:
        file_pattern (str): Glob pattern for file matching (e.g., '*.csv').
        data_loader (Callable[[Any], pd.DataFrame], optional): Function for loading 
            individual files. Defaults to pd.read_csv for CSVs, otherwise pd.read_json.
        show_progress (bool, optional): Show progress bar during file loading. Defaults to True.
        **load_options: Additional options passed to the data loader (like pd.read_csv or pd.read_json).

    Returns:
        pd.DataFrame: A single DataFrame combining the data from all matching files.

    Raises:
        FileNotFoundError: If no files match the given glob pattern.
    """
    
    # Choose the appropriate data loading function based on file extension if not provided
    if data_loader is None:
        if ".csv" in file_pattern.lower():
            data_loader = pd.read_csv
        else:
            data_loader = pd.read_json

    # Find all files matching the given pattern
    matched_files = glob(file_pattern)
    if not matched_files:
        raise FileNotFoundError(f"No files matched the pattern: {file_pattern}")

    if max_files is not None:
        max_index = max_files if max_files < len(matched_files) else len(matched_files)
        matched_files = matched_files[:max_index]

    # Load data from each file into a dataframe
    dataframes = []
    for file_path in tqdm(matched_files, disable=not pbar):
        df = data_loader(file_path, **load_options)
        dataframes.append(df)

    # Combine all loaded dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df

class Files:
    """A class to manage files with associated URLs and cache functionality."""
    
    def __init__(self, file_name: str, url: str = None, cache_dir: str = DEFAULT_CACHE_DIR):
        """
        Initialize a file object with a name, optional download URL, and cache directory.

        Args:
            file_name (str): Name of the file.
            url (str): URL to download the file from if not present.
            cache_dir (str): Directory to store cached files. Defaults to a global cache dir.
        """
        self.file_name = file_name
        self.url = url
        self.cache_dir = cache_dir
        self.file_path = os.path.join(self.cache_dir, file_name)

    def ensure_exists(self):
        """Ensure the file exists locally, downloading if necessary."""
        if not os.path.isfile(self.file_path):
            if self.url is None:
                raise FileDownloadError(f"No URL provided for {self.file_name}. Cannot download.")
            self.download_file()

    def download_file(self):
        """Download the file from the provided URL."""
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            print(f"Downloading {self.file_name} from {self.url}")
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

            with open(self.file_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192)):
                    if chunk:
                        f.write(chunk)
        except requests.RequestException as e:
            raise FileDownloadError(f"Failed to download {self.file_name}: {e}")

    def get_path(self):
        """Return the local path of the file, downloading if necessary."""
        self.ensure_exists()
        return self.file_path

class DataFiles(Files):
    """A class specifically for data files with predefined URLs."""
    a = "b"

