import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('yolo11x-seg.pt')

def get_segmentation_masks(image):
    height, width = image.shape[:2]
    results = model(image, stream=True)
    class_instances = []
    masks = []
    
    # Track instance count for each class
    instance_counts = {}
    
    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                class_name = result.names[int(cls)]
                
                # Update instance count
                if class_name not in instance_counts:
                    instance_counts[class_name] = 1
                else:
                    instance_counts[class_name] += 1
                
                # Create instance-specific label
                instance_label = f"{class_name}{instance_counts[class_name]}"
                
                # Convert mask to numpy and resize
                mask = seg.cpu().numpy()
                mask = cv2.resize(mask, (width, height))
                mask = (mask * 255).astype(np.uint8)
                
                class_instances.append(instance_label)
                masks.append(mask)
    
    return class_instances, masks

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