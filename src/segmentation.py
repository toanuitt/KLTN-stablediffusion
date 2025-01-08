import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')

def get_segmentation_masks(image):
    height, width = image.shape[:2]
    results = model(image, stream=True)
    classes = []
    masks = []
    
    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                # Convert mask to numpy and resize to match input image
                mask = seg.cpu().numpy()
                mask = cv2.resize(mask, (width, height))
                mask = (mask * 255).astype(np.uint8)
                classes.append(result.names[int(cls)])
                masks.append(mask)
    
    return classes, masks

def create_mask_for_class(image, masks, selected_class_idx):
    if selected_class_idx >= 0 and len(masks) > selected_class_idx:
        mask = masks[selected_class_idx]
        # Ensure mask and image have same dimensions
        height, width = image.shape[:2]
        mask = cv2.resize(mask, (width, height))
        
        mask_image = np.zeros_like(image)
        mask_binary = mask > 127
        mask_image[mask_binary] = 255
        return mask_image
    return None