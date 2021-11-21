import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Diabetes1.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 0:
        out = 'Non Diabetic'
    else: out = 'Diabetic'

    return render_template('index.html', prediction_text='Person is likely to be  :  {}'.format(out))

if __name__ == "__main__":
    app.run(debug=True)