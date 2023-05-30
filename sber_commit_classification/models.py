import os
import re
import nltk
import numpy as np
import pandas as pd
import pickle
from typing import List, Union, Tuple, Dict
from string import punctuation

from nltk.stem import WordNetLemmatizer

import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
import transformers

import torchmetrics as tm

import sber_commit_classification
from sber_commit_classification.unixcoder import UniXcoder
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, fbeta_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def show_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Метод для отображения метрик
    :param y_pred: предсказанные значения
    :param y_true: реальные значения
    """
    print('Train:')
    print('    Accuracy:', round(accuracy_score(y_pred, y_true), 3))
    print('    F1_score:', round(f1_score(y_pred, y_true), 3))
    print('    ROC_AUC:', round(roc_auc_score(y_pred, y_true), 3))
    print('    Precision:', round(precision_score(y_pred, y_true), 3))
    print('    Recall:', round(recall_score(y_pred, y_true), 3))
    print('    F2_score:', round(fbeta_score(y_pred, y_true, beta=2), 3))
    print('--------------------')


class SimpleCatboostClassificator:
    def __init__(self, is_bow=False):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.clf = CatBoostClassifier()
        self.path = os.path.dirname(sber_commit_classification.__file__)
        self.is_bow = is_bow
        if is_bow:
            self.clf.load_model(self.path + "\\models\\simpleCatboostClassificatorBoW")
        else:
            self.clf.load_model(self.path + "\\models\\simpleCatboostClassificatorTFIDF")
        nltk.download("stopwords")
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    @staticmethod
    def __check_data(data: pd.DataFrame):
        if type(data) == "<class 'pandas.core.frame.DataFrame'>":
            raise AttributeError(f"A pandas.core.frame.DataFrame was expected but a {type(data)} was received")
        columns_set = {'commit_message',
                       'files_changed',
                       'lines_inserted',
                       'lines_deleted',
                       'Imports added',
                       'Imports deleted'}
        if len(columns_set & set(data.columns)) != len(columns_set):
            raise KeyError(f"data's columns must contain only {list(columns_set)}")

    def __prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        enc_commit_mes = self.encode_text(data)
        return np.hstack([data.drop(columns=['commit_message']).values, enc_commit_mes])

    def my_tokenizer(self, text: str) -> List[str]:
        """
        Метод для разбиения текста на токены.
        :param text: Текст для токенизации
        :return: Список токенов
        """
        def is_good(word: str) -> bool:
            if re.fullmatch('[^' + punctuation + ']+', word):
                return True
            return False

        tokens = self.tokenizer.tokenize(text.lower())
        tokens = [word for word in tokens if is_good(word)]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def encode_text(self, data: pd.DataFrame) -> np.ndarray:
        """
        Метод для кодирования токенов в текст
        params:
            daya - DataFrame с данными
            tokenizer - токенайзер
            is_BoW - bool флаг для определения типа векторизатора
        returns:
            mes_train - закодированные commit message в train
            mes_val - закодированные commit message в val
        """
        tokenizer = self.my_tokenizer
        if self.is_bow:
            vocabulary = pickle.load(open(self.path + '\\vectorizers\\bow_vectorizer.pkl', 'rb'))
            vectorizer = CountVectorizer(tokenizer=tokenizer, vocabulary=vocabulary)
            encode_message = vectorizer.transform(data['commit_message'].values).toarray()
        else:
            vocabulary = pickle.load(open(self.path + '\\vectorizers\\tfidf_vectorizer.pkl', 'rb'))
            vectorizer = TfidfVectorizer(tokenizer=tokenizer, vocabulary=vocabulary)
            vectorizer.idf_ = np.load(self.path + '\\vectorizers\\idf.npy')
            encode_message = vectorizer.transform(data['commit_message'].values).toarray()
        return encode_message

    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        SimpleCatboostClassificator.__check_data(x_data)
        prep_data = self.__prepare_data(x_data)
        return self.clf.predict(prep_data)

    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        SimpleCatboostClassificator.__check_data(x_data)
        prep_data = self.__prepare_data(x_data)
        return self.clf.predict_proba(prep_data)


class CatboostWithCodeEncoding:
    def __init__(self, is_svd: bool = True, device: str = 'cpu'):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.unixcoder = UniXcoder("microsoft/unixcoder-base").to(device)
        self.roberta = RobertaModel.from_pretrained("roberta-base").to(device)
        self.unixcoder.eval()
        self.roberta.eval()

        self.device = device
        self.clf = CatBoostClassifier()
        self.path = os.path.dirname(sber_commit_classification.__file__)
        if is_svd:
            self.clf.load_model(self.path + "\\models\\catboostWithCodeEncodingSVD")
            self.decomposition = pickle.load(open(self.path + '\\models\\svd.pkl', 'rb'))
        else:
            self.clf.load_model(self.path + "\\models\\catboostWithCodeEncodingPCA")
            self.decomposition = pickle.load(open(self.path + '\\models\\pca.pkl', 'rb'))

    @staticmethod
    def __check_data(data: pd.DataFrame):
        if type(data) == "<class 'pandas.core.frame.DataFrame'>":
            raise AttributeError(f"A pandas.core.frame.DataFrame was expected but a {type(data)} was received")
        columns_set = {'commit_message',
                       'files_changed',
                       'lines_inserted',
                       'lines_deleted',
                       'Imports added',
                       'Imports deleted',
                       'file_new',
                       'file_past'}
        if len(columns_set & set(data.columns)) != len(columns_set):
            raise KeyError(f"data's columns must contain only {list(columns_set)}")

    def __prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        enc_mes = self.mes_enc(data)
        enc_code = self.code_enc(data)
        new_data = np.hstack([data.drop(columns=['commit_message', 'file_new', 'file_past']).values, enc_mes, enc_code])
        return self.decomposition.transform(new_data)

    def code_enc(self, data: pd.DataFrame) -> np.ndarray:
        res = []
        for i in range(len(data)):
            curr = data.iloc[i]['file_new']
            prev = data.iloc[i]['file_past']
            with torch.no_grad():
                curr_tokens_ids = self.unixcoder.tokenize([curr], max_length=512, mode="<encoder-only>")
                curr_tokens_ids = torch.tensor(curr_tokens_ids).to(self.device)
                curr_embedding = self.unixcoder(curr_tokens_ids)[1]

                prev_tokens_ids = self.unixcoder.tokenize([prev], max_length=512, mode="<encoder-only>")
                prev_tokens_ids = torch.tensor(prev_tokens_ids).to(self.device)
                prev_embedding = self.unixcoder(prev_tokens_ids)[1]

                res.append(torch.cat([curr_embedding, prev_embedding], dim=1))
        return torch.cat(res, dim=0).to('cpu').detach().numpy()

    def mes_enc(self, data: pd.DataFrame) -> np.ndarray:
        res = []
        for i in range(len(data)):
            sent = str(data.iloc[i]['commit_message'])
            with torch.no_grad():
                inputs = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=20,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True)

                mess_vec = self.roberta(input_ids=torch.tensor(inputs['input_ids']).view(1, -1).to(self.device),
                                        attention_mask=torch.tensor(
                                            inputs['attention_mask']
                                        ).view(1, -1).to(self.device),
                                        token_type_ids=torch.tensor(
                                            inputs["token_type_ids"]
                                        ).view(1, -1).to(self.device)
                                        )[0][:, 0].view(-1).to('cpu')
                res.append(mess_vec.detach().numpy())
        return np.vstack(res)

    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        CatboostWithCodeEncoding.__check_data(x_data)
        prep_data = self.__prepare_data(x_data)
        return self.clf.predict(prep_data)

    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray:
        CatboostWithCodeEncoding.__check_data(x_data)
        prep_data = self.__prepare_data(x_data)
        return self.clf.predict_proba(prep_data)


class SimpleCommitDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, device: str = 'cpu'):
        self.x = x
        self.y = y
        self.device = device

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.x[idx], device=self.device, dtype=torch.float),\
            torch.tensor(self.y[idx], device=self.device, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.x)


class SimpleSberNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(2310)
        self.fc1 = nn.Linear(2310, 128, bias=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 16, bias=True)
        self.bn2 = nn.BatchNorm1d(16)
        self.classifier = nn.Linear(16, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            self.relu(
                self.bn2(
                    self.fc2(
                        self.relu(
                            self.bn1(
                                self.fc1(
                                    self.bn0(x)
                                )
                            )
                        )
                    )
                )
            )
        )


class SimpleNNCommitClassifier(pl.LightningModule):
    def __init__(self, loss_func: nn.Module = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3])),
                 lr_func=None, is_pretrained: bool = False, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev = device

        # Models for data preprocessing
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.unixcoder = UniXcoder("microsoft/unixcoder-base").to(device)
        self.roberta = RobertaModel.from_pretrained("roberta-base").to(device)
        self.unixcoder.eval()
        self.roberta.eval()

        # Main parameters for fit
        self.lr_func = lr_func
        self.loss_func = loss_func
        self.optim = None

        self.path = os.path.dirname(sber_commit_classification.__file__)

        # Main func for metrics
        self.accuracy = tm.Accuracy(task='binary').to(device)
        self.f1 = tm.F1Score(task='binary').to(device)
        self.roc_auc = tm.AUROC(task='binary').to(device)

        self.clf = SimpleSberNet().to(device)
        if is_pretrained:
            self.clf.load_state_dict(torch.load(self.path + "\\models\\SimpleSberNet.pth"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(x)

    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        SimpleNNCommitClassifier.__check_data(x_data)
        prep_data = self.__prepare_data(x_data)
        ans = self.clf(torch.tensor(prep_data, dtype=torch.float32, device=self.dev)).to('cpu').detach().numpy()
        return np.argmax(ans, axis=1)

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray,
            val_x: pd.DataFrame, val_y: np.ndarray,
            optimizer: torch.optim.Optimizer,
            accelerator: str = 'cpu', max_epochs: int = 20,
            batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:

        SimpleNNCommitClassifier.__check_data(train_x)
        SimpleNNCommitClassifier.__check_data(val_x)
        if {0, 1} != set(train_y): raise AttributeError("train_y must contain only 0 or 1")
        if {0, 1} != set(val_y): raise AttributeError("val_y must contain only 0 or 1")

        prep_train_x = self.__prepare_data(train_x)
        prep_val_x = self.__prepare_data(val_x)
        self.optim = optimizer

        # Datasets
        train_dataset = SimpleCommitDataset(prep_train_x, train_y, device=self.dev)
        val_dataset = SimpleCommitDataset(prep_val_x, val_y, device=self.dev)

        # DataLoaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True)

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True)

        trainer = pl.Trainer(accelerator=accelerator,
                             max_epochs=max_epochs)
        trainer.fit(self, train_dataloader, val_dataloader)
        return train_dataloader, val_dataloader

    def code_enc(self, data: pd.DataFrame) -> np.ndarray:
        res = []
        for i in range(len(data)):
            curr = data.iloc[i]['file_new']
            prev = data.iloc[i]['file_past']
            with torch.no_grad():
                curr_tokens_ids = self.unixcoder.tokenize([curr], max_length=512, mode="<encoder-only>")
                curr_tokens_ids = torch.tensor(curr_tokens_ids).to(self.device)
                curr_embedding = self.unixcoder(curr_tokens_ids)[1]

                prev_tokens_ids = self.unixcoder.tokenize([prev], max_length=512, mode="<encoder-only>")
                prev_tokens_ids = torch.tensor(prev_tokens_ids).to(self.device)
                prev_embedding = self.unixcoder(prev_tokens_ids)[1]

                res.append(torch.cat([curr_embedding, prev_embedding], dim=1))
        return torch.cat(res, dim=0).to('cpu').detach().numpy()

    def mes_enc(self, data: pd.DataFrame) -> np.ndarray:
        res = []
        for i in range(len(data)):
            sent = str(data.iloc[i]['commit_message'])
            with torch.no_grad():
                inputs = self.tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    max_length=20,
                    pad_to_max_length=True,
                    return_token_type_ids=True,
                    truncation=True)

                mess_vec = self.roberta(input_ids=torch.tensor(inputs['input_ids']).view(1, -1).to(self.device),
                                        attention_mask=torch.tensor(
                                            inputs['attention_mask']
                                        ).view(1, -1).to(self.device),
                                        token_type_ids=torch.tensor(
                                            inputs["token_type_ids"]
                                        ).view(1, -1).to(self.device)
                                        )[1].view(-1).to('cpu')
                res.append(mess_vec.detach().numpy())
        return np.vstack(res)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer],
                                                                         List[torch.optim.lr_scheduler.LambdaLR]]]:
        if self.lr_func is None:
            return self.optim
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lr_func)
            return [self.optim], [scheduler]

    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        data, targets = train_batch

        preds = self.clf(data).view(-1)
        loss = self.loss_func(preds, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(preds, targets), prog_bar=True)
        self.log('train_f1', self.f1(preds, targets), prog_bar=True)
        self.log('train_roc_auc', self.roc_auc(preds, targets), prog_bar=True)
        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        data, targets = val_batch

        preds = self.clf(data).view(-1)
        loss = self.loss_func(preds, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, targets), prog_bar=True)
        self.log('val_f1', self.f1(preds, targets), prog_bar=True)
        self.log('val_roc_auc', self.roc_auc(preds, targets), prog_bar=True)

    @staticmethod
    def __check_data(data: pd.DataFrame):
        if type(data) == "<class 'pandas.core.frame.DataFrame'>":
            raise AttributeError(f"A pandas.core.frame.DataFrame was expected but a {type(data)} was received")
        columns_set = {'commit_message',
                       'files_changed',
                       'lines_inserted',
                       'lines_deleted',
                       'Imports added',
                       'Imports deleted',
                       'file_new',
                       'file_past'}
        if len(columns_set & set(data.columns)) != len(columns_set):
            raise KeyError(f"data's columns must contain only {list(columns_set)}")

    def __prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        enc_mes = self.mes_enc(data)
        enc_code = self.code_enc(data)
        new_data = np.hstack([data.drop(columns=['commit_message', 'file_new', 'file_past']).values, enc_mes, enc_code])
        return new_data


class MiddleCommitDataset(Dataset):
    def __init__(self, x: pd.DataFrame, y: np.ndarray,
                 tokenizer: transformers.models.roberta.tokenization_roberta.RobertaTokenizer,
                 mess_model: nn.Module, code_model: nn.Module,
                 max_len: int = 64):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.mess_model = mess_model
        self.code_model = code_model
        self.max_len = max_len

    def __getitem__(self, idx: int):
        sent = str(self.x['commit_message'][idx])
        inputs = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.int32)

        code_new = str(self.x['file_new'])
        code_past = str(self.x['file_past'])

        curr_tokens_ids = self.code_model.tokenize([code_new], mode="<encoder-only>", padding=True)
        curr_tokens_ids = torch.tensor(curr_tokens_ids)

        prev_tokens_ids = self.code_model.tokenize([code_past], mode="<encoder-only>", padding=True)
        prev_tokens_ids = torch.tensor(prev_tokens_ids)

        dct = {
            'main_data': torch.tensor(self.x.drop(['commit_message', 'file_new', 'file_past'],
                                                  axis=1).values[idx].reshape(-1)).type(dtype=torch.float),
            'inputs': inputs,
            'curr_tokens_ids': curr_tokens_ids.type(torch.int32),
            'prev_tokens_ids': prev_tokens_ids.type(torch.int32),
            'targets': torch.tensor(self.y[idx], dtype=torch.float32),
        }
        return dct

    def __len__(self):
        return len(self.x)


class MiddleSberNet(nn.Module):
    def __init__(self, is_full: bool):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 16, bias=True)
        self.bn2 = nn.BatchNorm1d(16)
        self.classifier = nn.Linear(16, 1, bias=True)

        self.code_model = UniXcoder("microsoft/unixcoder-base")
        self.mess_model = RobertaModel.from_pretrained("roberta-base")
        self.code_model.train()
        self.code_model.train()
        if not is_full:
            for param1, param2 in zip(self.code_model.parameters(), self.mess_model.parameters()):
                param1.requires_grad = False
                param2.requires_grad = False
            self.mess_model.pooler.dense = nn.Linear(768, 256)
            self.bn0 = nn.BatchNorm1d(1798)
            self.fc1 = nn.Linear(1798, 128, bias=True)
        else:
            self.bn0 = nn.BatchNorm1d(1798)
            self.fc1 = nn.Linear(2310, 128, bias=True)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        mess_vec = self.mess_model(input_ids=x['inputs']['input_ids'],
                                   attention_mask=x['inputs']['attention_mask'],
                                   token_type_ids=x['inputs']["token_type_ids"]
                                   )[1]
        main_data = x['main_data']
        prev_embedding = self.code_model(x['prev_tokens_ids'].view(main_data.size()[0], -1))[1]
        curr_embedding = self.code_model(x['curr_tokens_ids'].view(main_data.size()[0], -1))[1]

        new_x = torch.cat([main_data, mess_vec, prev_embedding, curr_embedding], dim=1)
        return self.classifier(
            self.relu(
                self.bn2(
                    self.fc2(
                        self.relu(
                            self.bn1(
                                self.fc1(
                                    self.bn0(new_x)
                                )
                            )
                        )
                    )
                )
            )
        )


class MiddleNNCommitClassifier(pl.LightningModule):
    def __init__(self, loss_func: nn.Module = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2])),
                 lr_func=None, is_full: bool = False, device: str = 'cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dev = device

        # Main parameters for fit
        self.optim = None
        self.lr_func = lr_func
        self.loss_func = loss_func

        self.path = os.path.dirname(sber_commit_classification.__file__)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Main func for metrics
        self.accuracy = tm.Accuracy(task='binary').to(device)
        self.f1 = tm.F1Score(task='binary').to(device)
        self.roc_auc = tm.AUROC(task='binary').to(device)

        self.clf = MiddleSberNet(is_full).to(device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.clf(x)

    def fit(self, train_x: pd.DataFrame, train_y: np.ndarray,
            val_x: pd.DataFrame, val_y: np.ndarray,
            optimizer: torch.optim.Optimizer,
            accelerator: str = 'cpu', max_epochs: int = 20,
            batch_size: int = 4, max_len: int = 64) -> Tuple[DataLoader, DataLoader]:
        MiddleNNCommitClassifier.__check_data(train_x)
        MiddleNNCommitClassifier.__check_data(val_x)
        if {0, 1} != set(train_y): raise AttributeError("train_y must contain only 0 or 1")
        if {0, 1} != set(val_y): raise AttributeError("val_y must contain only 0 or 1")

        self.optim = optimizer

        # Datasets
        train_dataset = MiddleCommitDataset(train_x,
                                            train_y,
                                            self.tokenizer,
                                            self.clf.mess_model,
                                            self.clf.code_model,
                                            max_len=max_len)
        val_dataset = MiddleCommitDataset(val_x,
                                          val_y,
                                          self.tokenizer,
                                          self.clf.mess_model,
                                          self.clf.code_model,
                                          max_len=max_len)

        # DataLoaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True)

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True)

        trainer = pl.Trainer(accelerator=accelerator,
                             max_epochs=max_epochs)
        trainer.fit(self, train_dataloader, val_dataloader)
        return train_dataloader, val_dataloader

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer],
                                                                         List[torch.optim.lr_scheduler.LambdaLR]]]:
        if self.lr_func is None:
            return self.optim
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lr_func)
            return [self.optim], [scheduler]

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx: int):
        dct = train_batch
        targets = dct['targets']

        preds = self.clf(dct).view(-1)
        loss = self.loss_func(preds, targets)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(preds, targets), prog_bar=True)
        self.log('train_f1', self.f1(preds, targets), prog_bar=True)
        self.log('train_roc_auc', self.roc_auc(preds, targets), prog_bar=True)
        return loss

    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int):
        dct = val_batch
        targets = dct['targets']

        preds = self.clf(dct).view(-1)
        loss = self.loss_func(preds, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, targets), prog_bar=True)
        self.log('val_f1', self.f1(preds, targets), prog_bar=True)
        self.log('val_roc_auc', self.roc_auc(preds, targets), prog_bar=True)

    @staticmethod
    def __check_data(data: pd.DataFrame):
        if type(data) == "<class 'pandas.core.frame.DataFrame'>":
            raise AttributeError(f"A pandas.core.frame.DataFrame was expected but a {type(data)} was received")
        columns_set = {'commit_message',
                       'files_changed',
                       'lines_inserted',
                       'lines_deleted',
                       'Imports added',
                       'Imports deleted',
                       'file_new',
                       'file_past'}
        if len(columns_set & set(data.columns)) != len(columns_set):
            raise KeyError(f"data's columns must contain only {list(columns_set)}")
