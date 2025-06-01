from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.keras")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
