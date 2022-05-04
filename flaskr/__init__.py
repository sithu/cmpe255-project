import os

from flask import Flask
from flask import redirect, url_for, jsonify


from ml.model import RegressionModel

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize ML Model
    print('init: ML Model...started')
    model = RegressionModel()
    print('init:ML Model is completed')

    @app.route('/')
    def index():
        return redirect(url_for('hello'))

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'


    @app.route('/predict/<feature_a>/<feature_b>')
    def predict(feature_a:int, feature_b:int):
        result = model.predict(feature_a, feature_b)
        return jsonify(result)


    return app