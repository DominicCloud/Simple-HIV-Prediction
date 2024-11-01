# app.py
from flask import Flask, request, render_template
from hiv_classifier import HIVClassifier

app = Flask(__name__)
classifier = HIVClassifier()

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {
        'Age': int(request.form['age']),
        'Marital Staus': request.form['marital_status'],
        'STD': request.form['std'],
        'Educational Background': request.form['education'],
        'HIV TEST IN PAST YEAR': request.form['hiv_test'],
        'AIDS education': request.form['aids_education'],
        'Places of seeking sex partners': request.form['sex_partner_place'],
        'SEXUAL ORIENTATION': request.form['sexual_orientation'],
        'Drug- taking': request.form['drug_taking']
    }

    # Pass input_data to classifier's predict function
    result = classifier.predict(input_data)

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
