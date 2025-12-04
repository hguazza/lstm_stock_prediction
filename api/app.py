"""
API Flask/FastAPI para servir predições do modelo LSTM.
"""

from flask import Flask, request, jsonify
# from fastapi import FastAPI, HTTPException
# from src.model import load_model
# from src.predict import predict
# from src.preprocessing import normalize_data

app = Flask(__name__)
# app = FastAPI()

# Carregar modelo (será carregado quando a API iniciar)
# model = load_model('api/model_lstm.h5')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de verificação de saúde da API.
    """
    return jsonify({"status": "healthy"}), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Endpoint para realizar predições.
    
    Recebe dados JSON e retorna predições do modelo.
    """
    try:
        # data = request.get_json()
        # predictions = predict(model, data)
        # return jsonify({"predictions": predictions.tolist()}), 200
        return jsonify({"message": "Endpoint de predição - implementar lógica"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

