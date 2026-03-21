import os
import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import medmnist
from medmnist import INFO

def stream_explode_npz_robust(npz_path, output_root):
    with zipfile.ZipFile(npz_path, 'r') as archive:
        files = archive.namelist()
        
        for split in ['train', 'val', 'test']:
            img_file = f'{split}_images.npy'
            lbl_file = f'{split}_labels.npy'
            if img_file not in files: continue

            split_dir = os.path.join(output_root, split)
            os.makedirs(split_dir, exist_ok=True)

            with archive.open(img_file) as f:
                version = np.lib.format.read_magic(f)
                shape, fort, dtype = np.lib.format.read_array_header_1_0(f)
                
                with archive.open(lbl_file) as lf:
                    labels = np.load(lf)

                img_size = np.prod(shape[1:]) * dtype.itemsize
                
                print(f"⚙️ Robustly extracting {shape[0]} images to {split_dir}...")
                
                for i in tqdm(range(shape[0])):
                    raw_data = f.read(img_size)
                    if not raw_data: break
                    
                    img_array = np.frombuffer(raw_data, dtype=dtype).reshape(shape[1:])
                    
                    img_path = os.path.join(split_dir, f"{i}_{labels[i][0]}.jpg")
                    
                    if os.path.exists(img_path):
                        continue
                        
                    try:
                        img = Image.fromarray(img_array)
                        with open(img_path, 'wb') as img_file_handle:
                            img.save(img_file_handle, "JPEG", quality=100)
                    except OSError:
                        time.sleep(0.1)
                        img.save(img_path, "JPEG", quality=100)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subsets = ['pathmnist', 'dermamnist', 'bloodmnist', 'octmnist']
    res = 224

    for flag in subsets:
        print(f"\n{'='*40}")
        print(f"Processing {flag}...")
        
        try:
            info = INFO[flag]
            DataClass = getattr(medmnist, info['python_class'])
            npz_file = os.path.join(current_dir, f"{flag}_224.npz")
            
            # 1. Download 
            if not os.path.exists(npz_file):
                print(f"Downloading {flag}...")
                try:
                    # DataClass with download=True downloads the file. 
                    # We catch potential memory errors when it tries to load automatically.
                    dataset = DataClass(split='train', download=True, size=res, root=current_dir)
                except Exception as e:
                    print(f"⚠️ Exception during loading data (but file should be downloaded): {e}")
            else:
                print(f"✅ {flag}_224.npz is already downloaded.")
            
            # 2. Extract
            out_folder = os.path.join(current_dir, f"{flag}_dataset")
            if os.path.exists(npz_file):
                stream_explode_npz_robust(npz_file, out_folder)
                print(f"🧹 Deleting {npz_file} to save space...")
                os.remove(npz_file)
            else:
                print(f"❌ Error: {npz_file} not found after download step.")

        except Exception as e:
            print(f"❌ Failed to process {flag}: {e}")

if __name__ == "__main__":
    main()
