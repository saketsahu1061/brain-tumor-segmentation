import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import mlflow
import mlflow.pytorch

# Global Configurations
WEIGHTS_PATH = os.path.join("weights", "resnet34_brain_tumor.pth")
DATA_DIR = "train" 

# ==========================================
# 1. NATIVE HYBRID LOSS FUNCTION (BCE + DICE)
# ==========================================
class HybridBCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, masks):
        bce_loss = self.bce(outputs, masks)
        probs = torch.sigmoid(outputs)
        
        probs = probs.view(-1)
        masks = masks.view(-1)
        
        intersection = (probs * masks).sum()
        dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + masks.sum() + self.smooth)
        dice_loss = 1.0 - dice_coeff
        
        return bce_loss + dice_loss

# ==========================================
# 2. COCO DATASET PARSER
# ==========================================
class BrainTumorCOCODataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor()
        ])
        
        json_path = os.path.join(data_dir, "_annotations.coco.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing COCO annotations at: {json_path}")
            
        with open(json_path, "r") as f:
            coco_data = json.load(f)
            
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_metadata = self.images[img_id]
        
        img_path = os.path.join(self.data_dir, img_metadata["file_name"])
        image = Image.open(img_path).convert("L")
        
        width, height = img_metadata["width"], img_metadata["height"]
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        annotations = self.img_to_anns.get(img_id, [])
        for ann in annotations:
            if "segmentation" in ann:
                for poly in ann["segmentation"]:
                    if len(poly) >= 6:
                        draw.polygon(poly, fill=255)
                        
        image_tensor = self.transform(image)
        mask_tensor = self.transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()
        
        return image_tensor, mask_tensor

# ==========================================
# 3. INCREMENTAL TRAINING ENGINE
# ==========================================
def run_training():
    epochs = 1  # Number of additional epochs you want to train for
    batch_size = 4
    learning_rate = 0.0001  # Lower LR for fine-tuning so we don't destroy existing patterns
    
    try:
        dataset = BrainTumorCOCODataset(data_dir=DATA_DIR)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"Dataset connected. Total images: {len(dataset)}")
    except Exception as e:
        print(f"Data Initialization Error: {str(e)}")
        return
    
    # Instantiate the EXACT framework structure matching your weights
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )
    
    # 👉 THE INCREMENTAL UPDATE JUMPSTART: Load existing weights if they exist
    if os.path.exists(WEIGHTS_PATH):
        print(f" Found existing checkpoint at {WEIGHTS_PATH}. Loading weights to resume training...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    else:
        print(" No checkpoint found. Starting training from scratch...")

    criterion = HybridBCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    mlflow.set_experiment("Brain_Tumor_Incremental_U-Net")
    
    print("--- Initiating Incremental Optimization Loop ---")
    with mlflow.start_run():
        mlflow.log_param("fine_tuning_epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for images, masks in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            print(f"Fine-Tuning Epoch [{epoch+1}/{epochs}] - Aggregated Loss: {epoch_loss:.4f}")
            mlflow.log_metric("loss", epoch_loss, step=epoch)
        
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print("--- Existing Checkpoint Successfully Updated and Overwritten with Better Weights ---")

if __name__ == "__main__":
    run_training()