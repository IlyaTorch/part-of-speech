from wtforms import Form, StringField


class NameForm(Form):
    sentence = StringField('Input a sentence')
