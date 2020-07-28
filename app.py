from flask import Flask, render_template, request
from forms import NameForm
from model.model_preparing import MODEL, WORD, TAG
from model.inference import predict_tags

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    predicted_tags = None
    if request.method == 'POST':
        sentence = ' ' + request.form['sentence']
        predicted_tags = predict_tags(MODEL, sentence, WORD, TAG)

    form = NameForm()

    return render_template('index.html', form=form, predicted_tags=predicted_tags)
