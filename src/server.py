from flask import Flask
from flask import render_template


app = Flask(__name__, template_folder='HTML')


@app.route('/')
def index():
    return render_template('index.html')
