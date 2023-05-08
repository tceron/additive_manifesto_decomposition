import numpy as np
import pandas as pd
import torch
from itertools import combinations
from utils import *
import glob
from sentence_transformers import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_distance_seq_by_pol_domain(mat, metadata, output_dir):

    dist_matrices = []
    pol_domains = set(list(metadata[:, 1]))

    for pol_domain in pol_domains:
        # print(map_label_name[int(pol_domain)])
        ind = [i[0] for i in np.argwhere(metadata == pol_domain)]
        embeddings = mat[ind,:]
        meta = metadata[ind,:]

        parties = list(set(list(meta[:,0])))

        p2idx = {p:i for i, p in enumerate(parties)}
        distance_matrix = np.zeros([len(parties), len(parties)])

        for p1, p2 in combinations(parties, 2):

            ind1, ind2 = [i[0] for i in np.argwhere(meta[:, 0] == p1)], [i[0] for i in np.argwhere(meta[:, 0] == p2)]
            mat1, mat2 = embeddings[ind1, :], embeddings[ind2, :]

            similarities = 1-util.cos_sim(mat1, mat2).numpy()
            distance = np.mean(similarities)

            distance_matrix[p2idx[p1], p2idx[p2]] = distance
            distance_matrix[p2idx[p2], p2idx[p1]] = distance

        dist_matrices.append(distance_matrix)
        pol_domain_df = pd.DataFrame(distance_matrix, columns=parties, index=parties)
        save_dataframe(pol_domain_df, output_dir, f"cosine_{pol_domain}")

    mean_distance = np.mean(dist_matrices, axis=0)
    df = pd.DataFrame(mean_distance, columns=parties, index=parties)
    print(df, "\n")
    save_dataframe(df, output_dir, "aggregated_cosine")

def compute_distances(lang):
    models_emb = glob.glob(f"./embeddings/{lang}/*/embeddings.p")

    for model_emb in models_emb:

        metadata = pickle.load(open(model_emb.replace("embeddings.p", "metadata.p"), "rb")).astype(str)
        embeddings = read_embeddings(model_emb)

        for _set in ["annotated", "predicted"]:
            if _set == "annotated":
                metadata[:, -1] = [map_labels[float(i)] for i in metadata[:, -1]]
                output_dir = f"./results/{lang}/annotated/" + model_emb.split("/")[-2]
            else:
                metadata[:, -1] = pickle.load(open("./classifier/predictions/bigram_xlm_roberta_all_german_speaking_epoch_2.p", "rb"))
                output_dir = f"./results/{lang}/predicted/" + model_emb.split("/")[-2]
            print(f"-- Model with {_set} labels: ", model_emb)
            compute_distance_seq_by_pol_domain(embeddings, metadata, output_dir)


if __name__ == '__main__':
    pol_domains = pickle.load(open(f"./grouping/labels_2021_13.p", "rb"))

    map_labels, label2idx= {}, {}
    for i, (k, v) in enumerate(pol_domains.items()):
        label2idx[k]=i
        for cat in v:
            map_labels[cat]=i
    label2idx["other"]=13

    compute_distances("de")


