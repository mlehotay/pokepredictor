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
            'city': [args.get('city')],
            'close_to_water': [args.get('close_to_water')],
            'weather': [args.get('weather')],
            'temperature': [args.get('temperature')],
            'day_of_week': [args.get('weather')],
            'local_time': [args.get('temperature')]
        })
    id = int(pipe.predict(data))

    df = pd.read_csv('data/pokedex.csv')

    return render_template(
        'result.html',
        pokemon=df.loc[id]['name'],
        image = df.loc[id]['img'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
