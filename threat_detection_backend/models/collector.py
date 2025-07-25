import os
import hashlib
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

def reassemble_file(chunk_dir: str, original_file_name: str, output_dir: str = None) -> Optional[str]:
    """
    Reassembles chunks back into the original file.

    Args:
        chunk_dir (str): The directory containing the chunk files.
        original_file_name (str): The expected name of the reassembled file.
        output_dir (str, optional): The directory where the reassembled file will be saved.
                                    Defaults to the parent directory of chunk_dir.

    Returns:
        Optional[str]: The path to the reassembled file if successful, None otherwise.
    """
    if not os.path.isdir(chunk_dir):
        logger.error(f"Chunk directory not found: {chunk_dir}")
        return None

    if output_dir is None:
        output_dir = os.path.dirname(chunk_dir)
    os.makedirs(output_dir, exist_ok=True)

    reassembled_file_path = os.path.join(output_dir, original_file_name)
    
    # Get all chunk files, sorted by part number
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith(original_file_name) and ".part" in f],
                         key=lambda x: int(x.split(".part")[-1]))

    if not chunk_files:
        logger.warning(f"No chunks found for {original_file_name} in {chunk_dir}")
        return None

    logger.info(f"Reassembling {len(chunk_files)} chunks for {original_file_name}...")

    try:
        with open(reassembled_file_path, 'wb') as f_out:
            for chunk_file_name in chunk_files:
                chunk_file_path = os.path.join(chunk_dir, chunk_file_name)
                with open(chunk_file_path, 'rb') as f_in:
                    f_out.write(f_in.read())
        logger.info(f"Successfully reassembled: {reassembled_file_path}")
        return reassembled_file_path
    except Exception as e:
        logger.error(f"Error reassembling file {original_file_name}: {e}")
        # Clean up incomplete file
        if os.path.exists(reassembled_file_path):
            os.remove(reassembled_file_path)
        return None

def get_file_md5(file_path: str) -> str:
    """Calculates the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

if __name__ == "__main__":
    # Example usage:
    # This part assumes you have run chunker.py and have dummy_model.bin_chunks directory
    dummy_file_name = "dummy_model.bin"
    chunk_directory = f"{dummy_file_name}_chunks"
    
    # Simulate chunks existing if chunker.py was run previously
    if not os.path.exists(chunk_directory):
        print(f"Please run chunker.py first to create '{chunk_directory}' for testing collector.py.")
    else:
        reassembled_path = reassemble_file(chunk_directory, dummy_file_name)
        if reassembled_path:
            print(f"Reassembled file at: {reassembled_path}")
            # You might want to compare MD5 hashes here if you stored the original hash
            # original_md5 = "..." # Get this from chunker.py output or separate storage
            # reassembled_md5 = get_file_md5(reassembled_path)
            # print(f"Reassembled MD5: {reassembled_md5}")
            # if original_md5 == reassembled_md5:
            #     print("MD5 hashes match! File integrity verified.")
            # else:
            #     print("MD5 hashes DO NOT match! File corruption detected.")
        else:
            print("File reassembly failed.")
