import os.path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

import fasttext
import gzip
import wget
import shutil
import pandas as pd
import torch
from pathlib import Path
import pickle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sbert_representations(sentences, type_model):
    print(f"Loading {type_model}...")
    model = SentenceTransformer(type_model)
    embeddings = model.encode(sentences, convert_to_tensor=True).cpu().detach().numpy()

    return embeddings

def bert_representations(sentences, type_model):
    print(f"Loading {type_model}...")
    model = AutoModel.from_pretrained(type_model, return_dict=True, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(type_model)

    embeddings = []
    for i in range(len(sentences)):
        inputs = tokenizer.encode_plus(sentences[i], return_tensors='pt').to(device)
        outputs = model(**inputs)

        layer_output = torch.stack(outputs.hidden_states[-2:]).sum(0).squeeze().cpu().detach().numpy()

        embeddings.append(np.mean(np.vstack(layer_output[1:-1, :]), axis=0))

    return embeddings

def fasttext_representations(sentences, lang):
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.bin.gz"
    if not os.path.exists(f"cc.{lang}.300.bin"):
        wget.download(url)
        with gzip.open(url.split("/")[-1], 'rb') as f_in:
            with open(url.split("/")[-1].replace(".gz", ""), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print(f"Loading fasttext model...")
    model = fasttext.load_model(url.split("/")[-1].replace(".gz", ""))

    embeddings=[]
    for sent in sentences:
        tokens=fasttext.tokenize(sent)
        embs =  np.vstack([model[t] for t in tokens]).mean(0)
        embeddings.append(embs)
    return embeddings

def get_embeddings(df, output_dir, lang, type_model=None):
    df = df.dropna()

    parties = np.array(df["party"].tolist())
    sentences = df["text"].tolist()
    categories = df.label.tolist()

    if "bert" in type_model:
        embeddings = bert_representations(sentences, type_model)
    elif "fasttext" in type_model:
        embeddings = fasttext_representations(sentences, lang)
    else:
        embeddings = sbert_representations(sentences, type_model)

    embeddings = np.vstack(embeddings)
    metadata = np.c_[parties, categories]
    print(embeddings.shape, metadata.shape)

    pickle.dump(embeddings, open(f"{output_dir}/embeddings.p", "wb"))
    pickle.dump(metadata, open(f"./{output_dir}/metadata.p", "wb"))


def get_representations_datasets():

    models = ["fasttext_emb", "paraphrase-multilingual-mpnet-base-v2",
              "bert-base-german-cased", "tceron/sentence-transformers-party-similarity-by-domain"
            ]

    for m in models:
        print(m)
        if "tceron" in m:
            name_m = m.split("/")[-1]
            output_dir = f"./{outdir}/{name_m}/"
        else:
            output_dir = f"./{outdir}/{m}/"
        print(output_dir)
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        get_embeddings(df, output_dir, lang, m)


if __name__ == '__main__':

    lang="de"
    outdir = f"embeddings/{lang}"
    df = pd.read_csv(f"./data/de/manifestos_2021.csv", index_col=0)  #these are the manifestos from the last elections with annotations (2017)
    df = df[1:]
    print("len dataset: ", len(df))
    get_representations_datasets()
