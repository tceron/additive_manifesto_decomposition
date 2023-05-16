import os
import json
from collections import Counter
from math import ceil
from random import shuffle, seed
import pandas as pd
import glob
import pickle
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import torch


class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim=768, out_dim=17, inner_dim=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, inner_dim)
        self.linear2 = torch.nn.Linear(inner_dim, out_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return self.linear2(x)


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def read_manifesto_concat(path, category2issue, issue2i):
    df = pd.read_csv(
        path, dtype='object'   # Do not convert labels to floats
    ).drop(columns=['eu_code'])
    df.columns = ['text', 'label']
    df.label = df.label.map(str)
    df.label = df.label.map(lambda l: l[:-2] if l.endswith('.0') else l)
    df['label_i'] = df.label.map(
        lambda l: category2issue.get(l, 'other')
    )
    # Now we combine sentences into pairs and try to predict
    # the label of the second one based on the concatenated text.
    # We ignore the label of the first sentence.
    records = []
    for i in range(1, df.shape[0]):
        records.append((
            df.text[i-1] + ' ' + df.text[i], df.label_i[i]
        ))
    df = pd.DataFrame.from_records(records, columns=[
        'text', 'label'
    ])
    df['label_i'] = df.label.map(lambda l: issue2i[l])
    return df


def read_manifesto_concat_test(path, category2issue, issue2i):
    df = pd.read_csv(path, dtype='object')
    df.label = df.label.map(str)
    df.label = df.label.map(lambda l: l[:-2] if l.endswith('.0') else l)
    df['label_i'] = df.label.map(
        lambda l: category2issue.get(l, 'other')
    )
    # Not all sentences in the test set are in order, so we
    # need to check if sentence_n's are consecutive
    records = []
    for i in range(1, df.shape[0]):
        if int(df.sentence_n[i]) - int(df.sentence_n[i-1]) == 1:
            records.append((
                df.text[i-1] + ' ' + df.text[i], df.label_i[i]
            ))
        else:  # No context
            records.append((
                df.text[i], df.label_i[i]
            ))
    df = pd.DataFrame.from_records(records, columns=[
        'text', 'label'
    ])
    df['label_i'] = df.label.map(lambda l: issue2i[l])
    return df


def train_epoch(epoch_n, epoch_train_steps, 
                train_data, batch_size, 
                tokeniser, model, classification_head,
                loss_function, optimiser, sbert=False):
                # loss_function, optimiser, scheduler):
    epoch_losses = torch.zeros(epoch_train_steps)
    for i in tqdm(range(epoch_train_steps), desc=f'Epoch {epoch_n+1}'):
        lo = i * batch_size
        hi = lo + batch_size
        batch = train_data['text'][lo:hi].to_list()
        labels = torch.tensor(train_data['label_i'][lo:hi].to_list()).cuda()
        tokenisation = tokeniser(batch, padding=True, truncation=True, return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
        model_outputs = model(**model_inputs).last_hidden_state
        if sbert:
            representations = mean_pooling(model_outputs, model_inputs['attention_mask'])
        else:
            # CLS embeddings
            representations = model_outputs[:, 0, :]
        logits = classification_head(representations)
        loss = loss_function(logits, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        epoch_losses[i] = loss.item()
    return epoch_losses.mean().item()


def validate_epoch(epoch_n, epoch_valid_steps, dev_data, batch_size,
                   tokeniser, model, classification_head, loss_function, sbert=False):
    epoch_losses = torch.zeros(epoch_valid_steps)
    epoch_accuracies = torch.zeros(epoch_valid_steps)
    predicted_labels_epoch = torch.zeros(dev_data.shape[0])
    for i in tqdm(range(epoch_valid_steps), desc=f'Epoch {epoch_n+1}, validation'):
        lo = i * batch_size
        hi = lo + batch_size
        batch = dev_data['text'][lo:hi].to_list()
        labels = torch.tensor(dev_data['label_i'][lo:hi].to_list()).cuda()
        tokenisation = tokeniser(batch, padding=True, truncation=True, return_tensors='pt')
        model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
        with torch.no_grad():
            model_outputs = model(**model_inputs).last_hidden_state
            if sbert:
                representations = mean_pooling(model_outputs, model_inputs['attention_mask'])
            else:
                # CLS embeddings
                representations = model_outputs[:, 0, :]
            logits = classification_head(representations)
            predicted_labels = torch.argmax(logits, dim=-1)
            predicted_labels_epoch[lo:hi] = predicted_labels.detach().cpu().flatten()
            loss = loss_function(logits, labels)
        epoch_losses[i] = loss.item()
        epoch_accuracies[i] = (predicted_labels == labels).sum().item() / len(batch)
    return epoch_losses.mean().item(), epoch_accuracies.mean().item(), predicted_labels_epoch


seed(42)
torch.manual_seed(42)

with open('label2issue.json', 'r', encoding='utf-8') as inp:
        label2issue = json.load(inp)
if not os.path.exists('issue2i.pickle') or not os.path.exists('i2issue.pickle'):
    all_issues = sorted(set(label2issue.values()))
    issue2i = {v: i for i, v in enumerate(all_issues)}
    issue2i['other'] = len(issue2i)
    with open('issue2i.pickle', 'wb') as out:
        pickle.dump(issue2i, out)
    i2issue = {v: i for i, v in issue2i.items()}
    with open('i2issue.pickle', 'wb') as out:
        pickle.dump(i2issue, out)
else:
    print('Loading precomputed mappings')
    with open('issue2i.pickle', 'rb') as inp:
        issue2i = pickle.load(inp)
    with open('i2issue.pickle', 'rb') as inp:
        i2issue = pickle.load(inp)

train_data = []
country_codes = ['at', 'ch_de']  # ['de'] for Germany only
for country in tqdm(country_codes, desc='Reading the data'):
    files = glob.glob(f'../data/{country}/*.csv')
    for f in tqdm(files, leave=False):
        df = read_manifesto_concat(f, label2issue, issue2i)
        train_data.append(df)
train_data = pd.concat(train_data)
print(train_data.shape)

test_data = read_manifesto_concat_test('../data/de/manifestos_2021.csv', label2issue, issue2i)
print(test_data.shape)

idx_arr = [i for i in range(train_data.shape[0])]
shuffle(idx_arr)

dev_data_boundary = len(idx_arr) // 10
dev_idx = idx_arr[: dev_data_boundary]
train_idx = idx_arr[dev_data_boundary :]
print(f'Train data size: {len(train_idx)}; dev data size: {len(dev_idx)}')
dev_data = train_data.iloc[dev_idx, :]
train_data = train_data.iloc[train_idx, :]

batch_size = 176

model_name = 'xlm-roberta-large'
# model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
sbert = 'mpnet' in model_name
tokeniser = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = torch.nn.DataParallel(model)
model.cuda()
classification_head = ClassificationHead(input_dim=768 if sbert else 1024, out_dim=len(i2issue))
classification_head = torch.nn.DataParallel(classification_head)
classification_head.cuda()

optimiser = torch.optim.AdamW(list(model.parameters()) + list(classification_head.parameters()), 
                              lr=0.00001)

n_epochs = 2
loss_function = torch.nn.CrossEntropyLoss()
epoch_train_steps = ceil(train_data.shape[0] / batch_size)
epoch_valid_steps = ceil(dev_data.shape[0] / batch_size)
epoch_test_steps = ceil(test_data.shape[0] / batch_size)
for epoch_n in range(n_epochs):
    epoch_train_loss = train_epoch(epoch_n, epoch_train_steps, 
                                   train_data, batch_size, 
                                   tokeniser, model, classification_head,
                                   loss_function, optimiser, sbert=sbert)
    epoch_dev_loss, epoch_dev_accuracy, _ = validate_epoch(epoch_n, epoch_valid_steps, 
                                                        dev_data, batch_size,
                                                        tokeniser, model, classification_head,
                                                        loss_function, sbert=sbert)
    epoch_test_loss, epoch_test_accuracy, epoch_test_predictions = validate_epoch(epoch_n, epoch_test_steps, 
                                                         test_data, batch_size,
                                                         tokeniser, model, classification_head,
                                                         loss_function, sbert=sbert)
    print(f'{epoch_train_loss=}, {epoch_dev_loss=}, {epoch_dev_accuracy=}')
    print(f'{epoch_test_loss=}, {epoch_test_accuracy=}')
    with open(f'bigram_predictions_xlm_roberta_no_germany_epoch_{epoch_n+1}.pickle', 'wb') as out:
        pickle.dump(epoch_test_predictions.numpy(), out)
    print()
