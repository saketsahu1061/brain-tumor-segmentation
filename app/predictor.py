import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BrainTumorPredictor:
    def __init__(self, weights_path, device="cpu"):
        self.device = torch.device(device)
        
        # Instantiate the exact 5-stage framework model matching your 20-epoch weights
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=1,
            classes=1
        )
        
        # Load the state dictionary smoothly with zero key or shape conflicts
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).float()
            return mask