import os
import hashlib
import logging
from typing import List

logger = logging.getLogger(__name__)

def chunk_file(file_path: str, chunk_size_mb: int = 80, output_dir: str = None) -> List[str]:
    """
    Divides a file into smaller chunks.

    Args:
        file_path (str): The path to the file to be chunked.
        chunk_size_mb (int): The maximum size of each chunk in megabytes.
        output_dir (str, optional): The directory where chunks will be saved.
                                    Defaults to a subdirectory named after the file in the same directory.

    Returns:
        List[str]: A list of paths to the created chunk files.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(file_path), f"{file_name}_chunks")
    
    os.makedirs(output_dir, exist_ok=True)

    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_files = []
    part_num = 0

    logger.info(f"Chunking file: {file_name} into {chunk_size_mb}MB chunks...")

    with open(file_path, 'rb') as f_in:
        while True:
            chunk = f_in.read(chunk_size_bytes)
            if not chunk:
                break
            
            chunk_file_name = f"{file_name}.part{part_num:04d}"
            chunk_file_path = os.path.join(output_dir, chunk_file_name)
            
            with open(chunk_file_path, 'wb') as f_out:
                f_out.write(chunk)
            
            chunk_files.append(chunk_file_path)
            part_num += 1
            logger.info(f"Created chunk: {chunk_file_name}")

    logger.info(f"Finished chunking {file_name}. Total chunks: {len(chunk_files)}")
    return chunk_files

def get_file_md5(file_path: str) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

if __name__ == "__main__":
    # Example usage:
    # Create a dummy file for testing
    dummy_file_path = "dummy_model.bin"
    dummy_file_size_mb = 200 # 200MB dummy file
    with open(dummy_file_path, 'wb') as f:
        f.write(os.urandom(dummy_file_size_mb * 1024 * 1024))
    print(f"Created dummy file: {dummy_file_path} ({dummy_file_size_mb}MB)")

    try:
        chunks = chunk_file(dummy_file_path, chunk_size_mb=80)
        print(f"Chunks created: {chunks}")
    except FileNotFoundError as e:
        print(e)
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
            print(f"Removed dummy file: {dummy_file_path}")
