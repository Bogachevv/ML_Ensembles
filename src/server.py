import time
import ensembles
import pandas as pd
import pathlib
from pydantic import BaseModel
from typing import Optional, Union, List, Tuple, Any, Literal
from py_singleton import singleton
from collections import OrderedDict


from flask import Flask, request, Response
from flask import render_template, redirect, url_for, jsonify, abort
from flask_bootstrap import Bootstrap

app = Flask(__name__, template_folder='HTML')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


class ModelRecord(BaseModel):
    # model: Union[ensembles.RandomForestMSE, ensembles.GradientBoostingMSE]
    model: Any
    target: str
    status: Literal['not_fit', 'fit'] = 'not_fit'


@singleton
class Models(object):
    def __init__(self):
        self.models = dict()

    def __getitem__(self, item: int) -> ModelRecord:
        return self.models[item]

    def __setitem__(self, key: int, value: ModelRecord):
        self.models[key] = value

    def keys(self):
        return self.models.keys()

    def __contains__(self, item: int):
        return item in self.models


@app.route('/', methods=['GET', 'POST'])
def model_settings():
    models = Models()

    if request.method == 'GET':
        return render_template('index.html')

    # serving POST request
    model_type = request.form['model_selection']
    n_estimators = request.form['n_estimators']
    max_depth_on = ('max_depth_on' in request.form)
    max_depth = request.form['max_depth'] if max_depth_on else None
    feature_subsample_size_auto = ('feature_subsample_size_auto' in request.form)
    feature_subsample_size = request.form['feature_subsample_size'] if not feature_subsample_size_auto else None
    learning_rate = request.form['learning_rate'] if model_type == 'GradientBoosting' else None
    dataset = request.files['dataset']
    target_name = request.form['select_target']

    try:
        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth is not None else None
        feature_subsample_size = int(feature_subsample_size) if feature_subsample_size is not None else None
        learning_rate = float(learning_rate) if model_type == 'GradientBoosting' else None
    except ValueError:
        return abort(400, {'message': f"Can't convert params to numerical type"})

    print(model_type, n_estimators, max_depth, max_depth_on, feature_subsample_size, feature_subsample_size_auto, target_name,
          sep='\n')

    print(request.files)

    if dataset.filename == '':
        abort(400)

    print(request.form)

    model_no = 0

    estimator = init_model(model_type, n_estimators, max_depth, feature_subsample_size, learning_rate)
    models[model_no] = ModelRecord(model=estimator, target=target_name)

    # the "uploads" folder needs protection against execution
    dataset.save(f'uploads/{model_no}_dataset.csv')

    return redirect(url_for('model', model_no=model_no))


@app.route('/model/<int:model_no>/gui', methods=['GET'])
def model(model_no: int):
    return render_template('model.html', model_no=model_no)


@app.route('/model/<int:model_no>/api/fit', methods=['POST'])
def model_fit(model_no: int):
    models = Models()

    print(models.keys())

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    estimator, target = models[model_no].model, models[model_no].target
    target = target.strip('"')

    path = pathlib.Path(f'uploads/{model_no}_dataset.csv')

    if not path.exists():
        return abort(404, {'message': f"Can't find file {str(path)}"})

    data = pd.read_csv(path)
    y = data[target]
    X = data.drop(columns=[target]).to_numpy()

    estimator.fit(X, y)

    models[model_no] = ModelRecord(model=estimator, target=target, status='fit')

    data = {'model_description': {}, 'train_score': {'MSE': -1, 'R2': -1}, 'target': target}

    return jsonify(data)


@app.route('/model/<int:model_no>/api/status', methods=['GET'])
def get_model_status(model_no: int):
    models = Models()

    xml = 'not_fit' if model_no not in models else models[model_no].status
    return Response(xml, mimetype='text/xml'), 200


@app.route('/model/<int:model_no>/api/columns_meta', methods=['GET'])
def get_columns_meta(model_no: int):
    models = Models()

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    path = pathlib.Path(f'uploads/{model_no}_dataset.csv')

    if not path.exists():
        return abort(404, {'message': f"Can't find file {str(path)}"})

    data = pd.read_csv(path)
    meta_inf = {col_nm: str(data[col_nm].dtype) for col_nm in data if col_nm != models[model_no].target}

    return jsonify(meta_inf)


@app.route('/model/<int:model_no>/api/predict_single', methods=['POST'])
def predict_single(model_no: int):
    models = Models()

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    features = request.json
    data = pd.DataFrame(features)
    estimator = models[model_no].model

    pred = estimator.predict(data)

    resp = {models[model_no].target: pred}

    return jsonify(resp)


@app.route('/model/<int:model_no>/api/validation_score', methods=['POST'])
def validation_score(model_no: int):
    models = Models()

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    estimator, target = models[model_no].model, models[model_no].target
    target = target.strip('"')

    path = pathlib.Path(f'uploads/{model_no}_validation.csv')
    dataset = request.files['dataset']
    dataset.save(path)

    if not path.exists():
        return abort(404, {'message': f"Can't find file {str(path)}"})

    data = {'model_description': {}, 'score': {'MSE': -1, 'R2': -1}, 'target': target}

    return jsonify(data)


def init_model(model_type, n_estimators, max_depth, feature_subsample_size, learning_rate):
    if model_type == 'RandomForest':
        estimator = ensembles.RandomForestMSE(
            n_estimators=n_estimators,
            max_depth=max_depth,
            feature_subsample_size=feature_subsample_size
        )
        return estimator
    elif model_type == 'GradientBoosting':
        estimator = ensembles.GradientBoostingMSE(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            feature_subsample_size=feature_subsample_size
        )
        return estimator

# TODO:
#    implement pd.Df in estimators
