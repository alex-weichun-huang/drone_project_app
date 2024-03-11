import os
import sys
import warnings
warnings.filterwarnings("ignore")
dir_path = os.path.join(os.path.dirname(__file__), "detectron2")
sys.path.append(dir_path)


import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision.ops import nms 
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def load_model(model_dir, model_name, device="cpu"):
    """ 
    Load model from model_dir/model_name.
    """
    # load config from yaml file
    cfg = get_cfg()
    yaml_file = os.path.join(model_dir, 'config.yaml')
    with open(yaml_file, 'r') as f:
        yaml_string = f.read()
        cfg = cfg.load_cfg(yaml_string)
        cfg.MODEL.WEIGHTS = os.path.join(model_dir, model_name)
        cfg.MODEL.DEVICE = device

    # build the model and load checkpoint
    model = build_model(cfg)
    _ = model.eval()
    checkpointer = DetectionCheckpointer(model)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    
    return model


def preprocess_image(pil_image, device="cpu"):
    """ 
    Preprocess image for inference.
    """
    tensor_image = torch.as_tensor(np.array(pil_image).transpose(2, 0, 1).astype("float32")).to(device)
    image_dict = {"image": tensor_image, "height": tensor_image.shape[1], "width": tensor_image.shape[2]}
    return image_dict


def nms_all_classes(instances, iou_thresh):
    """ 
    Apply non-maximum suppression to inference instances regardless of class.
    """
    valid_ind = nms(instances.pred_boxes.tensor, instances.scores, iou_thresh)
    instances.pred_boxes.tensor = instances.pred_boxes.tensor[valid_ind]
    instances.scores = instances.scores[valid_ind]
    instances.pred_classes = instances.pred_classes[valid_ind]
    return instances


def inference(model, pil_image, output_path=None, score_thresh=0.75, device="cpu"):
    """
    Perform inference on a single image.
    """
    np_image = np.array(pil_image)
    image_dict = preprocess_image(pil_image, device=device)
    
    with torch.no_grad():
        detections = model([image_dict])[0]['instances']
        detections = nms_all_classes(detections, 0.05)
        boxes = detections.get_fields()["pred_boxes"].tensor.cpu().numpy()
        scores = detections.get_fields()["scores"].cpu().numpy()
        for box, score in zip(boxes, scores):
            if score >= score_thresh:
                cv2.rectangle(np_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        PIL_image = Image.fromarray(np_image)  
        if output_path is not None:
            PIL_image.save(output_path)
    
    return PIL_image


def streamlit_app(model, device):
    """
    Streamlit application main function
    """
    st.title("Detect Marching Band Members!")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # add bounding boxes to the image
        box_image = inference(model, image, score_thresh=0.75, device=device)
        st.image(box_image, caption='Box Image', use_column_width=True)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("./assets/model", "model_ckpt.pth", device=device)
    streamlit_app(model, device)
