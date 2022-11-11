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
n_neighbors = 25


folder = pathlib.Path(f"/home/arty/condition/envPlot/data/virtualhome/")
# embeddings = (f"/home/arty/condition/envPlot/notebooks/test.pth")
labels = ("/home/arty/condition/envPlot/notebooks/test_BEHAVE_5_label.pth")

def map_umap():
    # path_mapper = folder / (f"virtualhome_{n_neighbors}.pkl")

    # if os.path.exists(path_mapper):
    #     with (path_mapper).open("rb") as f:
    #         mapper = joblib.load(f)
    # else:
    embeddings = torch.load("/home/arty/condition/envPlot/notebooks/test_BEHAVE_5.pth")
    mapper = umap.UMAP(n_neighbors=n_neighbors).fit(embeddings.cpu())
        # with (path_mapper).open("wb") as f:
        #     joblib.dump(mapper, f)
    label = torch.load(f"{labels}")
    # label = np.asarray(label)
    fig = umap.plot.points(mapper, labels=label, width=1200, height=1200)
    print(fig)
    plt.show()
    plt.savefig(f"behave_test_5_{n_neighbors}.png")


if __name__ == "__main__":
    map_umap()
