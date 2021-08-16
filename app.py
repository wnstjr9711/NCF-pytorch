from flask import Flask
from model_predict import *
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'neural collaborative filtering'


@app.route('/recommend/<uid>')
def get_recommendation(uid):
    uid = eval(uid)
    result = json.dumps({'prediction': list(get_prediction(uid)['aptId'])})
    return result


if __name__ == '__main__':
    app.run()
