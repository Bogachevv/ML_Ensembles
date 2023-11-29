import time

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
    if request.method == 'GET':
        return render_template('index.html')

    # serving POST request
    model_type = request.form['model_selection']
    n_estimators = request.form['n_estimators']
    max_depth_on = ('max_depth_on' in request.form)
    max_depth = request.form['max_depth'] if max_depth_on else None
    feature_subsample_size_auto = ('feature_subsample_size_auto' in request.form)
    feature_subsample_size = request.form['feature_subsample_size'] if not feature_subsample_size_auto else None
    dataset = request.files['dataset']
    target_name = request.form['select_target']

    print(model_type, n_estimators, max_depth, max_depth_on, feature_subsample_size, feature_subsample_size_auto, target_name,
          sep='\n')

    print(request.files)

    if dataset.filename == '':
        abort(400)

    print(request.form)

    model_no = 0

    # the "uploads" folder needs protection against execution
    dataset.save(f'uploads/{model_no}_dataset.csv')

    return redirect(url_for('model', model_no=model_no))


@app.route('/model/gui', methods=['GET'])
def model():
    return render_template('model.html')


@app.route('/model/fit', methods=['GET'])
def model_fit():
    model_no = request.args.get('model_no')
    MODELS[model_no] = 0

    time.sleep(5)

    return 'OK', 200


@app.route('/model/status', methods=['GET'])
def get_model_status():
    model_no = request.args.get('model_no')
    xml = 'fit' if model_no in MODELS else 'not_fit'
    return Response(xml, mimetype='text/xml'), 200
