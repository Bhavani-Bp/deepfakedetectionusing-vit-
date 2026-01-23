import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

class FaceExtractor:
    def __init__(self, device=None):
        """
        Initialize the FaceExtractor with MTCNN.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"FaceExtractor initialized on {self.device}")
        
        # Keep_all=False returns only the largest face, which is usually what we want for single-person deepfakes
        # select_largest=True ensures we focus on the main subject
        self.mtcnn = MTCNN(
            keep_all=False, 
            select_largest=True, 
            device=self.device,
            post_process=False,
            image_size=224, # Standard size for many models, can be resized later
            margin=20
        )

    def process_video(self, video_path, num_frames=10):
        """
        Extract faces from a video.
        
        Args:
            video_path (str): Path to the video file.
            num_frames (int): Target number of frames to extract (will sample evenly).
            
        Returns:
            list: List of PIL Integers (cropped faces).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("Error: Video has 0 frames or is corrupt.")
            cap.release()
            return []
            
        # Calculate indices to sample
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        faces = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in indices:
                # Convert BGR (OpenCV) to RGB (PIL/MTCNN)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Detect and crop
                # MTCNN call returns a tensor or list of cropped faces
                # modifying to return PIL image by using save_path=None and return_prob=False
                # But MTCNN forward directly returns tensor. 
                # We want the crop. Let's use the internal detect + crop methods if we want more control,
                # OR just let MTCNN return the tensor and we convert back to PIL for visualization/saving.
                # Simplest usage:
                
                face_tensor = self.mtcnn(frame_pil)
                
                if face_tensor is not None:
                    # Convert tensor back to PIL for consistency
                    # Tensor is (C, H, W) normalized to [-1, 1] usually if post_process=True
                    # But we set post_process=False, so it is [0, 255]? 
                    # Actually MTCNN docs say: if post_process=False, returns float tensor in [0, 255]
                    
                    face_np = face_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)
                    faces.append(Image.fromarray(face_np))
        
        cap.release()
        return faces

    def process_image(self, image_path):
        """
        Extract face from a single image.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            face_tensor = self.mtcnn(img)
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).int().numpy().astype(np.uint8)
                return Image.fromarray(face_np)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        return None
