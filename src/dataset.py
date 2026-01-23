import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Detection.
    Expects directory structure:
    root_dir/
        train/
            real/
            fake/
        valid/
            real/
            fake/
    
    Or uses a CSV to map paths relative to root_dir.
    """
    def __init__(self, root_dir, csv_file=None, transform=None, mode='train'):
        """
        Args:
            root_dir (str): Path to the dataset root (e.g., 'real_vs_fake/real-vs-fake').
            csv_file (str): Path to the CSV file (e.g., 'train.csv'). Optional if folder structure is standard.
            transform (callable, optional): Transform to apply to images.
            mode (str): 'train', 'valid', or 'test'. Used to filter CSV or folder.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data = []

        # If CSV is provided, use it
        if csv_file and os.path.exists(csv_file):
            logger.info(f"Loading dataset from CSV: {csv_file}")
            df = pd.read_csv(csv_file)
            # Filter if CSV contains all data (though typical Kaggle structure implies separate folders)
            # The provided train.csv seems to have 'path' like 'train/real/xxxxx.jpg'
            # We filter by checking if the path starts with the mode
            # But the user has train.csv, valid.csv, test.csv separate usually.
            
            # Let's assume the CSV passed matches the 'mode' or contains the relevant paths.
            # We'll check if the file exists.
            
            for _, row in df.iterrows():
                rel_path = row['path'] # e.g., 'train/real/31355.jpg'
                full_path = os.path.join(root_dir, rel_path)
                
                # Check if file exists (robustness)
                if os.path.exists(full_path):
                    label = 1 if row['label_str'] == 'fake' else 0 # Adjust based on CSV: label=1 is 'real' in the snippet shown!
                    # WAIT! Snippet said: label,label_str -> 1,real. 
                    # So 1=REAL, 0=FAKE? Or 0=Real, 1=Fake?
                    # "1,real" suggests 1 is the class ID for 'real'.
                    # Standard convention is 0 for Negative (Real) and 1 for Positive (Fake), but we must follow the data.
                    # Let's align with the CSV: label 1 = real. 
                    # However, usually Deepfake Detection treats "Fake" as the target class (1).
                    # Let's standardize: 0 = Real, 1 = Fake.
                    # If CSV says label=1 is Real, we should map it carefully.
                    # "label,label_str -> 1,real". So CSV has 1=Real.
                    # Let's check a fake row if possible. Usually binary.
                    # Let's trust label_str.
                    
                    target = 1 if row['label_str'].lower() == 'fake' else 0
                    self.data.append((full_path, target))
        
        else:
            # Fallback: Walk folders
            target_dir = os.path.join(root_dir, mode) if mode else root_dir
            logger.info(f"Loading dataset from folders: {target_dir}")
            
            for label_name in ['real', 'fake']:
                class_dir = os.path.join(target_dir, label_name)
                if not os.path.exists(class_dir):
                    continue
                    
                label = 1 if label_name == 'fake' else 0
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(class_dir, fname)
                        self.data.append((full_path, label))

        logger.info(f"Found {len(self.data)} images for {mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return adjacent image or handle gracefully? 
            # For simplicity, we err. In production, we'd skip.
            # Let's try to load the previous one
            return self.__getitem__((idx - 1) % len(self.data))

        if self.transform:
            image = self.transform(image)

        return image, label
