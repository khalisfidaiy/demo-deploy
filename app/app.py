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
    try: 
        data = request.json  # Receive JSON input 
        image = np.array(data["image"]).reshape(1, 28, 28) / 255.0  # Preprocess input 
        prediction = model.predict(image) 
        response = {"prediction": int(np.argmax(prediction))}  # Return highest probability digit 
        return jsonify(response) 
    except Exception as e: 
        return jsonify({"error": str(e)}) 
 
# Run Flask application 
if __name__ == "__main__": 
    app.run(debug=True)