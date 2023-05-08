import glob
import pandas as pd
from utils import compute_mantel
import matplotlib.pyplot as pp
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def correlation_matrices(ground_truth, folder):
    results=[]
    files = glob.glob(f"./{folder}/*/*/aggregated*.csv")

    df_gold = pd.read_csv(f"./data/{ground_truth}.csv", index_col=0)
    gold_dist_mat = df_gold.to_numpy()
    gold_parties = df_gold.columns

    for file in files:

        f = file.split("/")
        df = pd.read_csv(file, index_col=0)

        text_arr = df.reindex(gold_parties)  #reordering indexes to correspond with categorical distance matrix
        text_arr = text_arr[gold_parties].to_numpy()  # reordering columns

        r_corr, p_val = compute_mantel(gold_dist_mat, text_arr)

        results.append({"model":f[-2], "set": f[-3],
                        "metric":f[-1].replace(".csv", ""), "r":round(r_corr, 2), "pval":round(p_val, 4)})

    df = pd.DataFrame.from_dict(results)
    for s in set(df.set.tolist()):
        tmp = df[df["set"]==s]
        tmp = tmp.sort_values(by=['r'], ascending=False)
        print(tmp.head(40))
        print()

def plot_pc1(arr, parties, val, **kwargs):
    pp.plot(arr, np.zeros_like(arr) + val, 'x', **kwargs)
    for line in range(0,len(arr)):
        pp.text(arr[line]+0.001, val, parties[line],
        horizontalalignment='left', rotation=40,
        size='medium', color='black')
    pp.show()


def rile_scoring():
    scores = []
    df = pd.read_csv("./data/de/manifestos_2021.csv", index_col=0)
    parties = set(df.party.tolist())
    for party in parties:
        tmp = df[df["party"]==party]
        labels = tmp.label.tolist()
        le = len([i for i in labels if i in rile_cats["left"]])/len(tmp)
        ri = len([i for i in labels if i in rile_cats["right"]])/len(tmp)
        scores.append((party, ri-le))
    scores = pd.DataFrame(scores, columns=["party", "rile"])
    return scores.sort_values("party")


def run_pca(mat, n_comp, parties):
    pca = PCA(n_components=n_comp)
    pca_transformed = pca.fit_transform(mat)
    df = pd.DataFrame(pca_transformed)
    df["party"]=parties
    if n_comp == 1:
        df.columns=["pc1", "party"]
    else:
        df.columns=["pc1", "pc2", "party"]
    return df.sort_values("party")


def rile_correlation_mds(scores):
    models = ["fasttext_emb", "paraphrase-multilingual-mpnet-base-v2",
              "bert-base-german-cased", "sentence-transformers-party-similarity-by-domain"
              ]
    results = []
    for model in models:
        for _set in ["annotated", "predicted"]:
            issue_dist = pd.read_csv(f"./results/de/{_set}/{model}/aggregated_cosine.csv", index_col=0)
            issue_dist_mat = issue_dist.to_numpy()  #this is also a cosine matrix
            df_agg = run_pca(issue_dist_mat, 1, issue_dist.columns)

            score = pearsonr(scores, df_agg.pc1.tolist())  #rile_scores.rile.tolist()

            results.append((_set, model, "Pearson", score[0], score[1]))

    df = pd.DataFrame(results, columns=["set", "model", "corr_test", "corr", "pval"])
    for s in set(df.set.tolist()):
        tmp = df[df["set"]==s]
        print(tmp.sort_values("corr"))


if __name__ == '__main__':

    rile_cats = {"left": [103, 105, 106, 107, 202, 403, 404, 406, 412, 413, 504, 506, 701],
                 "right": [104, 201, 203, 305, 401, 402, 407, 414, 505, 601, 603, 605, 606]}

    print("\n--- Running Mantel correlation between the text similarity distance matrix and the label-based distance matrix.")
    correlation_matrices("de/ground_truth/euclidean_l1norm_manifesto_2021", "results/de")

    print("\n--- Running Pearson correlation between the MDS text similarity and the rile score.")
    rile_scores = rile_scoring()
    rile_correlation_mds(rile_scores.rile.tolist())