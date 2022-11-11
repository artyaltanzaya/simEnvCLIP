import torch
import h5py
import clip
import glob
import os
import numpy as np
import torchvision.transforms as T
import einops
import pickle

task = "all"
if task == "all":
    folder = "/data/arty/igibson/imitation_learning/*.hdf5"
else:
    folder = (f"/data/arty/igibson/imitation_learning/*{task}*")

device = "cuda"
model, preprocess = clip.load("ViT-B/32")
env = "behave"


def load(filename, data_str=['rgb']):
    h5f = h5py.File(filename, 'r')
    arrays = []
    for name in data_str:
        var = h5f[name][:]
        arrays.append(var)
    return arrays


def frame_to_clip(final_images):
    image = []
    for i, n in enumerate(final_images):
        clip_features = torch.empty(len(final_images), 512)
        n = n.reshape(1, 3, 224, 224)
        with torch.no_grad():
            image_features = model.encode_image(n.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image.append(image_features)
    clip_features = torch.cat(image)
    return clip_features


def extract_clip():
    count = 0
    clip_embs = []
    label_embs = []

    for i in glob.glob(folder):
        base_folder = os.path.basename(i)
        print(f"Going into {base_folder}")
        extract_images = torch.tensor(np.array(load(i)))
        extract_images = torch.flatten(
            extract_images, start_dim=0, end_dim=1)
        extract_images = einops.rearrange(
            extract_images, "n h w r -> n r h w")
        transform = T.Resize(size=(224, 224))
        transform_img = transform(extract_images)
        assert transform_img.size(
            dim=2) == 224, "Images are not transformed"  # [n, rgb, 224, 224]
        print("Images are transformed")
        final_clip = frame_to_clip(transform_img)
        clip_embs.append(final_clip)
        label = torch.full((final_clip.size(dim=0), 1), count)
        label_embs.append(label)
        # label_embs.append(count)
        count += 1
        print(f"{count} Transformed")
    all_t = torch.cat(clip_embs)
    all_l = torch.cat(label_embs)
    all_l = torch.flatten(all_l)
    torch.save(all_t, f"./data/{env}/all.pth")
    torch.save(all_l, f"./data/{env}/all_label.pth")
    print(f"Image Features: {all_t.shape}")
    print(f"Label: {all_l.shape}")


if __name__ == "__main__":
    extract_clip()
