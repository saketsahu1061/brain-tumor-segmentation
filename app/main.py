import os
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from app.preprocess import preprocess_image
from app.predictor import BrainTumorPredictor

WEIGHTS_PATH = os.path.join("weights", "resnet34_brain_tumor.pth")
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights file at production path: {WEIGHTS_PATH}")
    
    predictor = BrainTumorPredictor(weights_path=WEIGHTS_PATH, device="cpu")
    print(f"[{datetime.now().isoformat()}] --- Model Loaded Safely Into Memory ---")
    yield
    print(f"[{datetime.now().isoformat()}] --- Cleaning up server resources ---")

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="Production API gateway for ResNet34 U-Net medical image segmentation.",
    version="1.1.0",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def home_page():
    model_status = "ONLINE / ACTIVE" if predictor is not None else "OFFLINE / ERROR"
    status_color = "#4CAF50" if predictor is not None else "#F44336"
    hardware = str(predictor.device).upper() if predictor is not None else "N/A"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Tumor Segmentation Hub</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #121212;
                color: #E0E0E0;
                margin: 0;
                padding: 40px;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: #1E1E1E;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }}
            h1 {{ margin-top: 0; color: #FFFFFF; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            .status-badge {{
                display: inline-block;
                background-color: {status_color};
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }}
            .card {{
                background-color: #2D2D2D;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #BB86FC;
            }}
            .card h3 {{ margin: 0 0 10px 0; color: #BB86FC; }}
            ul {{ list-style-type: none; padding: 0; margin: 0; }}
            li {{ margin-bottom: 10px; font-size: 15px; }}
            strong {{ color: #FFFFFF; }}
            .btn {{
                display: block;
                text-align: center;
                background-color: #03DAC6;
                color: #000000;
                text-decoration: none;
                padding: 14px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                transition: background-color 0.2s;
                margin-top: 30px;
            }}
            .btn:hover {{ background-color: #018786; color: #FFFFFF; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Brain Tumor Segmentation Core</h1>
            <p>Production Inference Server Gateway & Workspace Console</p>
            
            <div class="grid">
                <div class="card">
                    <h3>🖥️ System Telemetry</h3>
                    <ul>
                        <li>Gateway Status: <span class="status-badge">{model_status}</span></li>
                        <li>Compute Engine: <strong>{hardware}</strong></li>
                        <li>Local Time: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></li>
                    </ul>
                </div>
                
                <div class="card">
                    <h3>🧬 Model Metadata</h3>
                    <ul>
                        <li>Architecture: <strong>ResNet34 U-Net (SMP)</strong></li>
                        <li>Target Matrix: <strong>640 &times; 640 &times; 1 (Grayscale)</strong></li>
                        <li>Output Profile: <strong>1-Channel Binary Segmentation Mask</strong></li>
                    </ul>
                </div>
            </div>
            
            <div class="card" style="border-left-color: #CF6679;">
                <h3>💾 Checkpoint Allocation</h3>
                <ul>
                    <li>Active Target Location: <code>{WEIGHTS_PATH}</code></li>
                    <li>File Detection Security Check: <strong>PASSED / SECURE</strong></li>
                </ul>
            </div>
            
            <a href="/docs" class="btn">🚀 Enter Interactive Testing Sandbox (/docs)</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")
    
    try:
        start_time = time.perf_counter()
        
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        predicted_mask = predictor.predict(input_tensor)
        
        execution_duration_ms = (time.perf_counter() - start_time) * 1000
        
        tumor_pixel_count = int(predicted_mask.sum().item())
        tumor_detected = tumor_pixel_count > 0
        
        total_canvas_pixels = 640 * 640
        tumor_spatial_ratio_percentage = round((tumor_pixel_count / total_canvas_pixels) * 100, 4)
        
        return {
            "metadata": {
                "filename": file.filename,
                "timestamp": datetime.now().isoformat(),
                "model_architecture": "ResNet34-U-Net",
                "input_dimensions": "640x640"
            },
            "telemetry": {
                "inference_latency_ms": round(execution_duration_ms, 2),
                "compute_hardware_device": str(predictor.device),
                "execution_status": "success"
            },
            "diagnostics": {
                "tumor_detected": tumor_detected,
                "segmented_area_pixels": tumor_pixel_count,
                "tumor_spatial_ratio_percentage": tumor_spatial_ratio_percentage,
                "clinical_severity_assessment": "Anomalous mass detected" if tumor_detected else "No structural abnormalities identified"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal production inference failure: {str(e)}"
        )