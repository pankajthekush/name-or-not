from flask import Flask, request, jsonify
from load_model import get_predicted_label

app = Flask(__name__)

# Define a route and handler function
@app.route('/')
def hello():
    return 'name-or-not'

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['input_text']
    if isinstance(input_text,list):
        predicted_labels = {text: get_predicted_label(text) for text in input_text}
    if isinstance(input_text,str):
        predicted_labels = {input_text: get_predicted_label(input_text)}
    return jsonify({'result': predicted_labels})    


# Run the Flask application if executed directly
if __name__ == '__main__':
    app.run()
