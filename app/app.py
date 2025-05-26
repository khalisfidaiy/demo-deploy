from flask import Flask, request, jsonify 
import tensorflow as tf 
import numpy as np 
 
# Load the trained model 
model = tf.keras.models.load_model("mnist_model.h5") 
 
# Initialize Flask app 
app = Flask(__name__) 
 
# Define a route for prediction 
@app.route("/predict", methods=["POST"]) 
def predict(): 
    data = request.json(force=True)  # Receive JSON input 
    image = np.array(data["instances"])  # Preprocess input 
    prediction = model.predict(image)
    return jsonify(predictions.tolist())

@app.route("/health", method)

if __name__ == "__main__": 
    app.run(debug=True)
