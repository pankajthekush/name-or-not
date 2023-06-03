from flask import Flask, request
from load_model import get_predicted_label

app = Flask(__name__)

# Define a route and handler function
@app.route('/')
def hello():
    return 'name-or-not'

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get('input_text')
    predicted_label = get_predicted_label(input_text)
    return str(predicted_label) 


# Run the Flask application if executed directly
if __name__ == '__main__':
    app.run()
