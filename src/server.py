from flask import Flask, request
from flask import render_template, redirect, url_for
from flask_bootstrap import Bootstrap

app = Flask(__name__, template_folder='HTML')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)


@app.route('/')
def index():
    return 'Index'


@app.route('/api/select', methods=['GET', 'POST'])
def model_selection():
    if request.method == 'GET':
        return render_template('model_selection.html')

    # serving POST request
    model_type = request.form['model_selection']

    return redirect(url_for('model_params', model_type=model_type))


@app.route('/api/params', methods=['GET', 'POST'])
def model_params():
    if request.method == 'GET':
        model_type = request.args.get('model_type')

        return render_template('model_params.html', model_type=model_type)

    # serving POST request

    model_type = request.args.get('model_type')
    n_estimators = request.form['n_estimators']
    max_depth = request.form['max_depth']
    max_depth_on = ('max_depth_on' in request.form)
    feature_subsample_size = request.form['feature_subsample_size']
    feature_subsample_size_auto = ('feature_subsample_size_auto' in request.form)

    print(model_type, n_estimators, max_depth, max_depth_on, feature_subsample_size, feature_subsample_size_auto, sep='\n')

    # return redirect(url_for('index'))
    return redirect(url_for(
        endpoint='select_dataset',
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_depth_on=max_depth_on,
        feature_subsample_size=feature_subsample_size,
        feature_subsample_size_auto=feature_subsample_size_auto
    ))


@app.route('/api/dataset', methods=['GET', 'POST'])
def select_dataset():
    return render_template('select_dataset.html')

