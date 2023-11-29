import time
import ensembles
import pandas as pd
import pathlib

from flask import Flask, request, Response
from flask import render_template, redirect, url_for, jsonify, abort
from flask_bootstrap import Bootstrap

app = Flask(__name__, template_folder='HTML')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


MODELS = {}
DATASETS = {}


@app.route('/', methods=['GET', 'POST'])
def model_settings():
    global MODELS

    if request.method == 'GET':
        return render_template('index.html')

    # serving POST request
    model_type = request.form['model_selection']
    n_estimators = request.form['n_estimators']
    max_depth_on = ('max_depth_on' in request.form)
    max_depth = request.form['max_depth'] if max_depth_on else None
    feature_subsample_size_auto = ('feature_subsample_size_auto' in request.form)
    feature_subsample_size = request.form['feature_subsample_size'] if not feature_subsample_size_auto else None
    learning_rate = request.form['learning_rate']
    dataset = request.files['dataset']
    target_name = request.form['select_target']

    try:
        n_estimators = int(n_estimators)
        max_depth = int(max_depth) if max_depth is not None else None
        feature_subsample_size = int(feature_subsample_size) if feature_subsample_size is not None else None
        learning_rate = float(learning_rate)
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
    MODELS[model_no] = (estimator, target_name)

    # the "uploads" folder needs protection against execution
    dataset.save(f'uploads/{model_no}_dataset.csv')

    return redirect(url_for('model', model_no=model_no))


@app.route('/model/gui', methods=['GET'])
def model():
    return render_template('model.html')


@app.route('/model/fit', methods=['GET'])
def model_fit():
    global MODELS

    model_no = int(request.args.get('model_no'))

    print(MODELS.keys())

    if model_no not in MODELS:
        return abort(404, {'message': f"Can't find model with number {model_no}"})

    estimator, target = MODELS[model_no]
    target = target.strip('"')

    path = pathlib.Path(f'uploads/{model_no}_dataset.csv')

    if not path.exists():
        return abort(404, {'message': f"Can't find file {str(path)}"})

    data = pd.read_csv(path)
    y = data[target]
    X = data.drop(columns=[target])

    estimator.fit(X, y)

    MODELS[model_no] = (estimator, target)

    return 'OK', 200


@app.route('/model/status', methods=['GET'])
def get_model_status():
    model_no = request.args.get('model_no')
    xml = 'fit' if model_no in MODELS else 'not_fit'
    return Response(xml, mimetype='text/xml'), 200


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
