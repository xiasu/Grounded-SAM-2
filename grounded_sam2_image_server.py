import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from flask import Flask, request, jsonify
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# Flask app setup
app = Flask(__name__)

# Hyperparameters
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# Build Grounding DINO model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# Utility to convert single mask to RLE
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# Flask route for image segmentation
@app.route('/segment', methods=['POST'])
def segment_image():
    # Ensure request has files and text prompt
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Image file and text prompt are required."}), 400

    # Load image and text prompt
    image_file = request.files['image']
    text_prompt = request.form['prompt']
    print(text_prompt)

    # Save uploaded image temporarily
    img_path = "temp_image.jpg"
    image_file.save(img_path)

    # Load and preprocess the image
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    # Perform prediction using Grounding DINO
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # Process the box prompt for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Use SAM2 to generate masks
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Convert masks to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Convert mask to RLE format
    #mask_rles = [single_mask_to_rle(mask) for mask in masks]
    mask_rles = [mask.astype(int).tolist() for mask in masks]
    #print(len(mask_rles))
    # Prepare response data
    response = {
        "annotations": [
            {
                "class_name": label,
                "bbox": box.tolist(),
                "segmentation": mask_rle,
                "score": float(score)
            }
            for label, box, mask_rle, score in zip(labels, input_boxes, mask_rles, scores.tolist())
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h
    }
    #print(response)
    # Remove temporary image file
    os.remove(img_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
