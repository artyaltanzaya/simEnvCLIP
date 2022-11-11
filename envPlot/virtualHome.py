import torch
import clip
import glob
import os
import numpy as np
import torchvision.transforms as T
import einops
import pickle
from PIL import Image

device = "cuda"

folder = "/home/arty/condition/envPlot/data/virtualhome/Output/*"

model, preprocess = clip.load("ViT-B/32")

def frame2clip():
    image = []
    label_embs = []
    count = 0
    for i in glob.glob(folder):
        base_folder = os.path.basename(i)
        print(f"Going into {base_folder}")

        for f in glob.glob(f'/home/arty/condition/envPlot/data/virtualhome/Output/{base_folder}/*.png'):
            with Image.open(f) as im:
                image.append(im)
                image_input = torch.stack([preprocess(frame) for frame in image]).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            print(f"Image Features: {image_features.shape}")
            torch.save(image_features, "/home/arty/condition/envPlot/data/virtualhome/Output/clip_virtual.pth")
            label_embs.append(count)
            torch.save(label_embs, "./label.pth")
        count +=1
        print(count)


if __name__ == "__main__":
    frame2clip()
