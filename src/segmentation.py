import numpy as np
import cv2
from ultralytics import YOLO
import gradio as gr

# Initialize with None - will be set from app.py
model = None
ALLOWED_CLASSES = None
config = None  # Will be set from app.py


def initialize_yolo(yolo_opts):
    global model, ALLOWED_CLASSES, config
    model = YOLO(yolo_opts["model"]["weights"])
    ALLOWED_CLASSES = set(yolo_opts["classes"]["allowed"])
    config = yolo_opts


def get_segmentation_masks(image):
    height, width = image.shape[:2]
    results = model(
        image,
        conf=config["model"]["confidence"],
        iou=config["model"]["iou"],
        stream=True,
    )
    class_instances = []
    masks = []

    # Track instance count for each class
    instance_counts = {}

    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                class_name = result.names[int(cls)].lower()

                # Skip if class not in allowed classes
                if class_name not in ALLOWED_CLASSES:
                    continue

                # Update instance count
                if class_name not in instance_counts:
                    instance_counts[class_name] = 1
                else:
                    instance_counts[class_name] += 1

                # Create instance-specific label
                instance_label = f"id_{instance_counts[class_name]}"

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


def detect_objects(image):
    global stored_masks
    classes, masks = get_segmentation_masks(image)
    stored_masks = masks
    return gr.Dropdown(choices=classes)


def apply_mask_to_image(image, mask):
    if mask is None or image is None:
        return None
    # Create black background
    masked_image = np.zeros_like(image)
    # Copy only the masked region from original image
    masked_image[mask > 127] = image[mask > 127]
    return masked_image


def update_mask(image, selected_class_idx):
    if selected_class_idx is None:
        return None, None
    mask = create_mask_for_class(image, stored_masks, selected_class_idx)
    masked_region = apply_mask_to_image(image, mask)
    return mask, masked_region
