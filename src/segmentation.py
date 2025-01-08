from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('yolo11x-seg.pt')
def get_segmentation_masks(image):
    results = model(image, stream=True)
    classes = []
    masks = []
    
    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                mask = seg.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                classes.append(result.names[int(cls)])
                masks.append(mask)
    
    return classes, masks

def create_mask_for_class(image, masks, selected_class_idx):
    if selected_class_idx >= 0:
        mask = masks[selected_class_idx]
        mask_image = np.zeros_like(image)
        mask_image[mask > 127] = 255
        return mask_image
    return None