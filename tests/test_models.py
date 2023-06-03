import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

import tests
import sber_commit_classification as scc
import torch


def test_simple_catboost_classificator_bow():
    clf = scc.SimpleCatboostClassificator(is_bow=True)
    test_data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_data.csv',
                            index_col=0).drop(columns=['target', 'file_new', 'file_past'])
    pred = clf.predict(test_data)
    pred_proba = clf.predict_proba(test_data)
    true_pred = np.array([1., 0., 0., 0., 0.])
    true_pred_proba = np.around(np.array([[0.19930344, 0.80069656],
                                          [0.62785676, 0.37214324],
                                          [0.77815301, 0.22184699],
                                          [0.58733057, 0.41266943],
                                          [0.59577288, 0.40422712]]),
                                decimals=8)
    assert np.array_equal(pred, true_pred), 'False predict'
    assert np.array_equal(np.around(pred_proba, decimals=8), true_pred_proba), 'False predict_proba'


def test_simple_catboost_classificator_tfidf():
    clf = scc.SimpleCatboostClassificator(is_bow=False)
    test_data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_data.csv',
                            index_col=0).drop(columns=['target', 'file_new', 'file_past'])
    pred = clf.predict(test_data)
    pred_proba = clf.predict_proba(test_data)
    true_pred = np.array([1., 0., 0., 0., 0.])
    true_pred_proba = np.around(np.array([[0.20991665, 0.79008335],
                                          [0.647917, 0.352083],
                                          [0.7769106, 0.2230894],
                                          [0.56872527, 0.43127473],
                                          [0.63529126, 0.36470874]]),
                                decimals=8)
    assert np.array_equal(pred, true_pred), 'False predict'
    assert np.array_equal(np.around(pred_proba, decimals=8), true_pred_proba), 'False predict_proba'


def test_catboost_with_code_encoding_svd():
    clf = scc.CatboostWithCodeEncoding(is_svd=True)
    test_data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_data.csv',
                            index_col=0).drop(columns=['target'])
    pred = clf.predict(test_data)
    pred_proba = clf.predict_proba(test_data)
    true_pred = np.array([1., 0., 0., 1., 0.])
    true_pred_proba = np.around(np.array([[0.30724405, 0.69275595],
                                          [0.73467489, 0.26532511],
                                          [0.76042002, 0.23957998],
                                          [0.30409635, 0.69590365],
                                          [0.66289177, 0.33710823]]),
                                decimals=8)
    assert np.array_equal(pred, true_pred), 'False predict'
    assert np.array_equal(np.around(pred_proba, decimals=8), true_pred_proba), 'False predict_proba'


def test_catboost_with_code_encoding_pca():
    clf = scc.CatboostWithCodeEncoding(is_svd=False)
    test_data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_data.csv',
                            index_col=0).drop(columns=['target'])
    pred = clf.predict(test_data)
    pred_proba = clf.predict_proba(test_data)
    true_pred = np.array([1., 0., 0., 1., 0.])
    true_pred_proba = np.around(np.array([[0.38324248, 0.61675752],
                                          [0.6888583, 0.3111417],
                                          [0.70575683, 0.29424317],
                                          [0.38732384, 0.61267616],
                                          [0.72197256, 0.27802744]]),
                                decimals=8)
    assert np.array_equal(pred, true_pred), 'False predict'
    assert np.array_equal(np.around(pred_proba, decimals=8), true_pred_proba), 'False predict_proba'


def test_simple_sber_module():
    clf = scc.SimpleSberModule()
    optim = torch.optim.Adam(clf.parameters(), lr=0.001)

    data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_fit_data.csv', index_col=0).head(20)
    x = data.drop(['target'], axis=1)
    y = data['target']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=228, stratify=y)
    x_train.reset_index(inplace=True)
    x_val.reset_index(inplace=True)
    try:
        clf.fit(x_train, y_train.values, x_val, y_val.values, optim, max_epochs=1)
        clf.predict(x_val)
        assert True
    except Exception as e:
        assert False, str(e)


def test_simple_sber_module_pretrained():
    clf = scc.SimpleSberModule(is_pretrained=True)
    optim = torch.optim.Adam(clf.parameters(), lr=0.001)

    data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_fit_data.csv', index_col=0).head(20)
    x = data.drop(['target'], axis=1)
    y = data['target']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=228, stratify=y)
    x_train.reset_index(inplace=True)
    x_val.reset_index(inplace=True)
    try:
        clf.fit(x_train, y_train.values, x_val, y_val.values, optim, max_epochs=1)
        clf.predict(x_val)
        assert True
    except Exception as e:
        assert False, str(e)


def test_strong_sber_module():
    clf = scc.StrongSberModule()
    optim = torch.optim.Adam(clf.parameters(), lr=0.001)

    data = pd.read_csv(os.path.dirname(tests.__file__) + '\\data\\test_fit_data.csv', index_col=0).head(20)
    x = data.drop(['target'], axis=1)
    y = data['target']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=228, stratify=y)
    x_train.reset_index(inplace=True)
    x_val.reset_index(inplace=True)
    try:
        clf.fit(x_train, y_train.values, x_val, y_val.values, optim, max_epochs=1)
        assert True
    except Exception as e:
        assert False, str(e)
