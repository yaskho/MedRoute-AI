import h5py
import json
import shutil
import sys

def remove_key_recursive(d, target_key):
    """Recursively search and remove a key from a nested dictionary/list structure."""
    if isinstance(d, dict):
        if target_key in d:
            del d[target_key]
        for k, v in d.items():
            remove_key_recursive(v, target_key)
    elif isinstance(d, list):
        for item in d:
            remove_key_recursive(item, target_key)

def clean_h5_model(input_path, output_path):
    # 1. Create an exact byte-for-byte copy of the HDF5 file
    shutil.copy(input_path, output_path)
    print(f"[*] Created exact copy: {output_path}")

    # 2. Open the new file in read/write mode ('r+')
    with h5py.File(output_path, 'r+') as f:
        if 'model_config' in f.attrs:
            # Read model_config (could be bytes or string depending on h5py version)
            config_data = f.attrs['model_config']
            if isinstance(config_data, bytes):
                config_str = config_data.decode('utf-8')
            else:
                config_str = config_data

            # Parse the JSON architecture layout
            config_dict = json.loads(config_str)

            # Recursively strip 'quantization_config' from layers/configs
            remove_key_recursive(config_dict, 'quantization_config')

            # Convert back to JSON string
            new_config_str = json.dumps(config_dict)

            # Save it back with matching data type format (bytes vs string)
            if isinstance(config_data, bytes):
                f.attrs['model_config'] = new_config_str.encode('utf-8')
            else:
                f.attrs['model_config'] = new_config_str

            print("[+] Successfully removed 'quantization_config' from model_config metadata.")
        else:
            print("[-] Warning: 'model_config' attribute not found in the root group of this HDF5 file.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_h5.py <input_model.h5> <output_model.h5>")
    else:
        clean_h5_model(sys.argv[1], sys.argv[2])