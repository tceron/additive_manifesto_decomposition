from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from sentence_transformers import SentenceTransformer

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import argparse

def read_new_issues():
    issues = pickle.load(open("../grouping/labels_2021_13.p", "rb"))
    map_labels,label2name, name2label={}, {}, {}
    for i, (k, v) in enumerate(issues.items()):
        label2name[i]=k
        name2label[k]=i
        for cat in v:
            map_labels[cat]=i
        name2label["other"]=13
        label2name[13]="other"
    return map_labels, label2name, name2label

def load_model(st_model):
    model = SentenceTransformer(st_model, device='cuda')
    return model

def load_data(args):
    train_df = pd.read_csv(f"./data/{args.data_dir}/train.csv", index_col=0)
    train_df = train_df[["text", "label"]]
    # train_df["label"] = [map_labels[i] for i in train_df.label.tolist()]

    # Load the test dataset into a pandas dataframe.
    eval_df = pd.read_csv("./data/test.csv", index_col=0)
    eval_df = eval_df[["text", "label"]]
    eval_df["label"]=[map_labels[i] for i in eval_df.label.tolist()]
    print(eval_df)

    x_eval = eval_df["text"].values.tolist()
    y_eval = eval_df["label"].values.tolist()
    x_train = train_df["text"].values.tolist()
    y_train = train_df["label"].values.tolist()

    return x_train, x_eval, y_train, y_eval

def logistic_reg(x_train, x_eval, y_train, y_eval, file_name, multiclass_type="multinomial"):

    print("--- Multiclass:", multiclass_type, "with C: ")
    classifier = LogisticRegression(multi_class=multiclass_type,
                                 max_iter=1000, random_state=args.seed).fit(x_train, y_train)
    preds = classifier.predict(x_eval)

    print('Accuracy: ', accuracy_score(y_eval, preds))
    pickle.dump(preds, open(f"{output_dir}/{file_name}.p", "wb"))


def concanate_vectors(sentences, model):
    new_embs=[]
    encod_sent = model.encode(sentences)
    for i, sent in enumerate(encod_sent):
        if i != 0:
            new_embs.append(np.concatenate([encod_sent[i - 1], sent]))

def concanate_sentences(sentences):
    new_sents = []
    for i, sent in enumerate(sentences):
        if i != 0:
            new_sents.append(sentences[i - 1]+" "+sent)
    return new_sents

def encode_sentences(sentences, model):
    encod_sents = model.encode(sentences)
    print(encod_sents.shape)

    return encod_sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="german_speaking")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    st_model = "paraphrase-multilingual-mpnet-base-v2"

    map_labels, label2name, name2label = read_new_issues()

    for data in ["german_speaking", "germany"]:
        print(data)

        args.data_dir = data
        output_dir = f"./outputs/{args.data_dir}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x_train, x_eval, y_train, y_eval = load_data(args)
        y_train = y_train[1:]
        y_eval = y_eval[1:]

        model = load_model(st_model)
        x_train, x_eval = concanate_sentences(x_train), concanate_sentences(x_eval)
        X_train =encode_sentences(x_train, model)
        X_eval = encode_sentences(x_eval, model)

        logistic_reg(X_train, X_eval, y_train, y_eval, "preds_logistic")