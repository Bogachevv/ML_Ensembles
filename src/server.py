import ensembles
import pandas as pd
import pathlib
from typing import Optional, Union, List, Tuple, Any, Literal
from sklearn.model_selection import train_test_split

from flask import Flask, request, Response
from flask import render_template, redirect, url_for, jsonify, abort
from flask_bootstrap import Bootstrap

from model_storage import ModelRecord, Models


app = Flask(__name__, template_folder='HTML')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


# <------ Pages ------>


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
    test_size = request.form['test_size'] if 'test_size' in request.form else None

    try:
        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth is not None else None
        feature_subsample_size = int(feature_subsample_size) if feature_subsample_size is not None else None
        learning_rate = float(learning_rate) if model_type == 'GradientBoosting' else None
        if test_size is not None:
            test_size = int(test_size)
            test_size = test_size / 100 if test_size > 0 else None
    except ValueError:
        return abort(400, {'message': f"Can't convert params to numerical type"})

    if n_estimators < 1:
        return abort(400, {'message': f"n_estimators must be greater than 0"})
    if (max_depth is not None) and (max_depth < 1):
        return abort(400, {'message': "max_depth must be greater than 0"})
    if (feature_subsample_size is not None) and (feature_subsample_size < 1):
        return abort(400, {'message': "feature_subsample_size must be greater than 0"})
    if (learning_rate is not None) and (learning_rate <= 0):
        return abort(400, {'message': "learning_rate must be greater than 0"})
    if (test_size is not None) and ((test_size < 0) or (test_size > 1)):
        return abort(400, {'message': "test_size must be in [0, 100] %"})

    if dataset.filename == '':
        abort(400)
    if dataset.filename.rsplit('.', 1)[1].lower() != 'csv':
        return abort(415, {'message': f"Bad filename format"})

    model_no = models.next_model_no()

    estimator = init_model(model_type, n_estimators, max_depth, feature_subsample_size, learning_rate)
    models[model_no] = ModelRecord(model=estimator, target=target_name, test_size=test_size)

    try:
        # the "uploads" folder needs protection against execution
        dataset.save(f'uploads/{model_no}_dataset.csv')
    except Exception:
        return abort(500, {'message': f"Can't save dataset"})

    return redirect(url_for('model', model_no=model_no))


@app.route('/model/<int:model_no>/gui', methods=['GET'])
def model(model_no: int):
    return render_template('model.html', model_no=model_no)


@app.route('/model/<int:model_no>/fitting_curve', methods=['GET'])
def fitting_curve(model_no: int):
    return render_template('fitting_curve.html', model_no=model_no)


# <------ API ------>


@app.route('/model/<int:model_no>/api/fit', methods=['POST'])
def model_fit(model_no: int):
    models = Models()

    print(models.keys())

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    if models[model_no].status == 'fitting':
        return abort(409, 'Model is fitting now. Please wait a few minutes and try to get model state')

    models[model_no].status = 'fitting'

    estimator, target, test_size = models[model_no].model, models[model_no].target, models[model_no].test_size
    target = target.strip('"')

    path = pathlib.Path(f'uploads/{model_no}_dataset.csv')

    if not path.exists():
        return abort(404, {'message': f"Can't find file {str(path)}"})

    data = pd.read_csv(path)
    y = data[target]
    X = data.drop(columns=[target])
    meta_info = X.dtypes
    features = list(X.columns)
    X = X.to_numpy()
    y = y.to_numpy()

    print(meta_info)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        estimator.fit(X_train, y_train)
        train_score = estimator.calc_score(X_test, y_test)
    else:
        estimator.fit(X, y)
        train_score = estimator.calc_score(X, y)

    evaluated_on_test = (test_size is not None)
    train_score = {'MSE': train_score[0], 'R2': train_score[1], 'evaluated_on_test': evaluated_on_test}

    estimators_sp, mse_sp, rs2_sp = estimator.get_fit_curve(X, y)

    models[model_no] = ModelRecord(model=estimator, target=target, test_size=test_size, status='fit',
                                   meta_info=meta_info, features=features, train_score=train_score,
                                   fit_curve=(estimators_sp, mse_sp, rs2_sp)
                                   )

    description = fill_description(estimator)

    data = {'model_description': description, 'train_score': train_score, 'target': target}

    return jsonify(data)


@app.route('/model/<int:model_no>/api/status', methods=['GET'])
def get_model_status(model_no: int):
    models = Models()

    xml = 'not_fit' if model_no not in models else models[model_no].status
    return Response(xml, mimetype='text/xml'), 200


@app.route('/model/<int:model_no>/api/train_score', methods=['GET'])
def get_model_train_score(model_no: int):
    models = Models()

    score = models[model_no].train_score

    return jsonify(score)


@app.route('/model/<int:model_no>/api/description', methods=['GET'])
def get_model_description(model_no: int):
    models = Models()
    estimator = models[model_no].model

    description = fill_description(estimator)

    return jsonify(description)


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


@app.route('/model/<int:model_no>/api/predict', methods=['POST'])
def predict(model_no: int):
    models = Models()

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})
    model_rec = models[model_no]

    features = request.json

    print(f"DEBUG: predict {features=}")

    try:
        df = pd.DataFrame([features])
        df = df.astype(model_rec.meta_info)
        print(df)
        print(df.dtypes)
    except ValueError:
        return abort(422, {'message': f"Incorrect input type"})

    estimator = model_rec.model

    pred = estimator.predict(df.to_numpy())[0]

    resp = {'target': models[model_no].target, 'value': pred}

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

    dataset = pd.read_csv(path)
    y = dataset[target]
    X = dataset.drop(columns=[target]).to_numpy()

    score = estimator.calc_score(X, y)

    data = {'model_description': {}, 'score': {'MSE': score[0], 'R2': score[1]}, 'target': target}

    return jsonify(data)


@app.route('/models/<int:model_no>/api/fitting_curve', methods=['GET'])
def get_fitting_curve(model_no: int):
    models = Models()

    if model_no not in models:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    fit_curve, test_size = models[model_no].fit_curve, models[model_no].test_size

    estimators_sp, mse_sp, rs2_sp = fit_curve
    estimators_sp = list(map(int, estimators_sp))
    mse_sp = list(map(float, mse_sp))
    rs2_sp = list(map(float, rs2_sp))

    print(estimators_sp, mse_sp, rs2_sp)

    evaluated_on_test = (test_size is not None)
    data = {'estimators_count': estimators_sp,
            'score': {
                'MSE': mse_sp,
                'R2': rs2_sp},
            'evaluated_on_test': evaluated_on_test,
            }

    return jsonify(data)


def fill_description(estimator: Union[ensembles.RandomForestMSE, ensembles.GradientBoostingMSE]) -> dict:
    strategy = 'RandomForest' if isinstance(estimator, ensembles.RandomForestMSE) else 'GradientBoosting'
    max_deep = estimator.max_depth if estimator.max_depth is not None else 'unlimited'
    subsample_size = estimator.feature_subsample_size if estimator.feature_subsample_size is not None else 'auto'

    description = {
        'strategy': strategy,
        'ensembles_cnt': estimator.n_estimators,
        'max_deep': max_deep,
        'subsample_size': subsample_size,
    }

    if strategy == 'GradientBoosting':
        description['learning_rate'] = estimator.learning_rate

    return description


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
