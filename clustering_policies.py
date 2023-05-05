import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from utils import *
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from pathlib import Path


def plot_dendrogram(model, categories, **kwargs):

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=categories, **kwargs)



def compute_distance_seq_with_issue(mat, metadata):

    issues = set(list(metadata[:, 1]))

    categories = set(list(metadata[:, 1]))
    c2idx = {p: i for i, p in enumerate(categories)}
    distance_matrix = np.zeros([len(categories), len(categories)])

    for cat1, cat2 in combinations(issues, 2):
        # print(map_label_name[int(issue)])
        ind1 = [i[0] for i in np.argwhere(metadata == cat1)]
        ind2 = [i[0] for i in np.argwhere(metadata == cat2)]
        embeddings1 = mat[ind1,:]
        embeddings2 = mat[ind2,:]

        similarities = util.cos_sim(embeddings1, embeddings2).numpy()

        distance = np.mean(1-similarities)
        distance_matrix[c2idx[cat1], c2idx[cat2]] = distance
        distance_matrix[c2idx[cat2], c2idx[cat1]] = distance

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="average")

    model = model.fit(distance_matrix)
    plt.figure(figsize=(20, 10))
    labels = [map_labels[i] for i in list(c2idx.keys())]
    plt.title("Hierarchical Clustering of CMP Categories")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, labels, truncate_mode="level", leaf_rotation=90, p=len(labels))
    plt.xlabel("categories")

    Path("./data/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("./data/plots/hirarchical_withen.jpeg")
    plt.show()

    linkage_matrix = linkage(distance_matrix, method="average")

    dct = dict([(ind, {i}) for ind, i in enumerate(categories)])
    for i, row in enumerate(linkage_matrix, distance_matrix.shape[0]):
        dct[i] = dct[row[0]].union(dct[row[1]])
        del dct[row[0]]
        del dct[row[1]]
        if len(list(dct.values()))>= 11 and len(list(dct.values()))<=18:
            print(f"\n---Inspect {len(list(dct.values()))} clusters: \n")
            for i in range(len(list(dct.values()))):
                print(f"-Cluster {i}:")
                for c in list(dct.values())[i]:
                    print(map_labels[c])
                print("\n")


def compute_distances():

    # model_emb = f"./embeddings/{lang}/paraphrase-multilingual-mpnet-base-v2/embeddings.p"

    # embs = read_embeddings(model_emb, do_whiten=do_whiten)
    # metadata = pickle.load(open("./embeddings/de/metadata.p", "rb"))

    df = pd.read_csv("./data/de/manifestos_2021.csv", index_col=0)
    df=df.dropna()
    df = df[df["label"]!="H"]
    dic = dict(df.label.value_counts())
    exclude = [k for k,v in dic.items() if v < 10]
    print("\n-- Excluded categories from the clustering because they contained fewer than 10 examples: ", exclude, "\n")
    df=df[~df["label"].isin(exclude)]
    sentences = [sentence.replace("•", "") for sentence in df.text.tolist()]
    embs = sbert_representations(sentences, "paraphrase-multilingual-mpnet-base-v2")
    metadata = np.c_[df.party.tolist(), df.label.tolist()]

    kernel, bias = compute_kernel_bias(embs)
    norm_embs = transform_and_normalize(embs, kernel=kernel, bias=bias)

    compute_distance_seq_with_issue(norm_embs, metadata)


if __name__ == '__main__':
    df = pd.read_csv("./data/codebook_categories_MPDS2020a.csv")
    map_labels = dict(zip([str(i) for i in df.code], df.title))
    compute_distances()