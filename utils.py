import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mantel
from sentence_transformers import util, SentenceTransformer
from collections import defaultdict


def plot_cosine_matrix(matrix, name_f_plot, title_, yaxis, xaxis, output_dir):

    fig, ax = plt.subplots()

    matrix = np.tril(matrix).astype('float')
    matrix[matrix == 0] = 'nan'  # or use np.nan

    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            c = matrix[j, i]
            if str(c)!="nan":
                ax.text(i, j, str(round(c, 2)), va='center', ha='center')  #party_pair[i], party_pair[j], str(float(c)),

    plt.title(title_)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels([i for i in xaxis])
    ax.set_yticklabels([i for i in yaxis])

    plt.show()
    Path(f"{output_dir}/plots").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{output_dir}/plots/{name_f_plot}.jpeg")
    plt.close()


# def save_cosine_between_sentences(cosine_scores, sentences, party_pair, name_f, output_dir):
#     # Find the pairs with the highest cosine similarity scores
#     Path(f"{output_dir}/cos_sentences").mkdir(exist_ok=True, parents=True)
#     f_out = open(f"{output_dir}/cos_sentences/{name_f}.txt", "w")
#     pairs = []
#     for i in range(len(cosine_scores) - 1):
#         for j in range(i + 1, len(cosine_scores)):
#             pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
#
#     # Sort scores in decreasing order
#     pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
#     for pair in pairs:
#         i, j = pair['index']
#         f_out.write(f"Sent1 of party {party_pair[i].upper()}= {sentences[i]}\nSent2 of party {party_pair[j].upper()}= {sentences[j]}\nScore= {pair['score']}\n\n")
#     f_out.close()

def compute_kernel_bias(vecs, k=None):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    Code taken from: https://github.com/bojone/BERT-whitening
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    if k:
        return W[:,:k], -mu
    else:
        return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    Code taken from: https://github.com/bojone/BERT-whitening
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def calculate_sim_score(cos_results):
    sim_scores = []
    for party_pair in cos_results.keys():
        numerator=np.mean(cos_results[party_pair]["across_parties"]+cos_results[party_pair]["across_parties_t"])
        sim_score = numerator/np.mean(cos_results[party_pair]["party1"]+cos_results[party_pair]["party2"])
        sim_scores.append((party_pair[0], party_pair[1], sim_score))
        sim_scores.append((party_pair[1], party_pair[0], sim_score))
    return sim_scores


def categorical_distance_matrices_claims():
    dist_cat={}
    for dataset in ["macov", "wom"]:
        df = pd.read_csv(f"./data/categorical_distance/{dataset}_claim.csv")
        dist_cat[f"part_{dataset}_claim"]=df.party.tolist()
        df = 1-df.drop(columns=["party"]).to_numpy()
        dist_cat[f"dist_{dataset}_claim"] = df
    return dist_cat

def compute_mantel(cat_arr, text_arr):
    r, pval, z = mantel.test(cat_arr, text_arr, perms=10000, method='spearman', tail='two-tail')
    return r, pval

def l2_normalize(v):
    norm = np.sqrt(np.sum(np.square(v)))
    return v / norm

def save_dataframe(df, output_dir, file_name):
    Path(f"{output_dir}").mkdir(exist_ok=True, parents=True)
    df.to_csv(f"{output_dir}/" + str(file_name) +".csv")


def read_embeddings(path, do_whiten=True):
    with open(path, 'rb') as inp:
        embeddings = pickle.load(inp)
    if do_whiten:
        print("--- Whitening vectors ---")
        kernel, bias = compute_kernel_bias(embeddings)
        embeddings_norm = transform_and_normalize(embeddings, kernel, bias)
        return embeddings_norm.astype(float)
    else:
        return embeddings.astype(float)

def sbert_representations(sentences, type_model):
    print(f"Loading {type_model}...")
    model = SentenceTransformer(type_model)
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().detach().numpy()

    return embeddings




