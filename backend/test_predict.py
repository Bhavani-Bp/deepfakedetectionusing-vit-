import os
import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import DeepfakeEfficientNet, DeepfakeViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'deepfake_efficientnet.pth'))
model = DeepfakeEfficientNet(num_classes=2).to(DEVICE)
model.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from src.face_extraction import FaceExtractor
extractor = FaceExtractor(device=DEVICE)

video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'videoplayback.mp4'))

if os.path.exists(video_path):
    print(f"Testing video: {video_path}")
    faces = extractor.process_video(video_path, num_frames=10)
    print(f"Extracted {len(faces)} faces.")
    for i, face in enumerate(faces):
        tensor = transform(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            print(f"Frame {i} - Real: {probs[0,0].item():.4f}, Fake: {probs[0,1].item():.4f}")
else:
    print(f"Video not found: {video_path}")
