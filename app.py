import pickle


from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__, template_folder='templates')
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    args = request.form
    data = pd.DataFrame({
            'latitude': [float(args.get('latitude'))],
            'longitude': [float(args.get('longitude'))],
            'close_to_water': [args.get('close_to_water')],
            'city': [args.get('city')],
            # 'weather': [args.get('weather')],
            'temperature': [args.get('temperature')]
        })
    prediction = pipe.predict(data)
    return render_template(
        'result.html',
        prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
