import umap
import os
import pathlib
import joblib
import pickle
import umap.plot
import torch
import matplotlib.pyplot as plt
import numpy as np

device = "cuda"
env = "behave"
task = "*.hdf5"
sample_size = 50000
n_neighbors = 10


folder = pathlib.Path(f"/home/arty/condition/envPlot/data/{env}/")
embeddings = (f"/home/arty/condition/envPlot/data/{env}/all_{task}.pth")
labels = (f"/home/arty/condition/envPlot/data/{env}/all_label_{task}.pth")
# text_label = open('/home/arty/condition/envPlot/label.pkl', 'rb')
# text_label = pickle.load(text_label)


def map_umap():
    path_mapper = folder / (f"behave_{sample_size}{n_neighbors}.pkl")

    if os.path.exists(path_mapper):
        with (path_mapper).open("rb") as f:
            mapper = joblib.load(f)
    else:
        # embeddings = torch.load(folder / (f"all_{task}.pth"))
        # obs_mapper = umap.UMAP().fit(embeddings)
        embeddings = np.load(folder / "behavior.npy")
        print("Embeddings shape before sample", embeddings.shape)
        embeddings = embeddings[:sample_size, :]
        mapper = umap.UMAP(n_neighbors=n_neighbors).fit(embeddings)
        with (path_mapper).open("wb") as f:
            joblib.dump(mapper, f)
    label = torch.load(f"{labels}")
    label = label[:sample_size]
    print("Label shape", label.shape)
    fig = umap.plot.points(mapper, labels=label, width=1200, height=1200)
    print(fig)
    plt.show()
    plt.savefig(f"{task}_{env}_{sample_size}_{n_neighbors}.png")


if __name__ == "__main__":
    map_umap()
