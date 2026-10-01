from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('models/cyber_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = ""
    confidence = 0

    if request.method == 'POST':
        duration = int(request.form['duration'])
        protocol = int(request.form['protocol'])
        service = int(request.form['service'])
        src_bytes = int(request.form['src_bytes'])
        dst_bytes = int(request.form['dst_bytes'])
        wrong_fragment = int(request.form['wrong_fragment'])
        urgent = int(request.form['urgent'])
        hot = int(request.form['hot'])
        failed_logins = int(request.form['failed_logins'])

        features = np.array([[duration, protocol, service,
                              src_bytes, dst_bytes,
                              wrong_fragment, urgent,
                              hot, failed_logins]])

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        confidence = round(max(probability[0]) * 100, 2)

        if prediction[0] == 1:
            result = "Attack Detected"
        else:
            result = "Normal Traffic"

    return render_template('predict.html',
                           result=result,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
