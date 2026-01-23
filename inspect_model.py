import torch
import os

MODEL_PATH = 'models/deepfake_efficientnet.pth'

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
else:
    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        print("Model loaded.")
        
        # Check keys 
        keys = list(state_dict.keys())
        print(f"Total keys: {len(keys)}")
        
        # Find classifier weights
        for key in keys:
            if 'classifier' in key or 'fc' in key or 'head' in key:
                print(f"Key: {key}, Shape: {state_dict[key].shape}")
                
    except Exception as e:
        print(f"Error inspecting model: {e}")
