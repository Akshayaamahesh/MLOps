from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and its metadata
bundle = joblib.load('wine_model.joblib')
model = bundle['model']
feature_names = bundle['feature_names']
class_names = bundle['class_names']
example = bundle['example']


def pretty(name):
    return name.replace('od280/od315_of_diluted_wines', 'OD280/OD315').replace('_', ' ').title()


@app.route('/')
def home():
    return "Welcome to the Wine Cultivar Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form
            input_data = np.array([[float(data[name]) for name in feature_names]])

            # Perform the prediction
            probabilities = model.predict_proba(input_data)[0]
            predicted_class = class_names[int(np.argmax(probabilities))]

            return jsonify({
                "predicted_class": predicted_class,
                "probabilities": {c: round(float(p), 3) for c, p in zip(class_names, probabilities)},
            })
        except KeyError as e:
            return jsonify({"error": f"Missing feature: {e.args[0]}"}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return render_template(
        'predict.html',
        features=[(name, pretty(name), value) for name, value in zip(feature_names, example)],
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
