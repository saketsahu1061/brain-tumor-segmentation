import io 
import torch 
from PIL import Image 
import torchvision.transforms as T
 
def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    """
    Transforms raw uploaded user file bytes into an optimized 4D tensor 
    ready for native ResNet-34 U-Net inference.
    
    Args:
        file_bytes (bytes): Raw binary data stream from the HTTP request payload.
        
    Returns:
        torch.Tensor: A normalized 4D floating-point tensor of shape [1, 1, 640, 640].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    
    transform_pipeline = T.Compose([
        T.Resize((640, 640)),             
        T.ToTensor()  # Automatically scales pixels to a clean [0.0, 1.0] range
    ])
    
    tensor = transform_pipeline(img)
    tensor = tensor.unsqueeze(0)  # Add batch dimension -> [1, 1, 640, 640]
    return tensor