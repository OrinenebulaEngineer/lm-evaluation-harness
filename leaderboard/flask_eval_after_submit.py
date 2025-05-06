from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    model = data.get('model')
    print(f"🚀 Evaluation triggered for model: {model}")

    return jsonify({"status": "evaluation started", "model": data.get("model")})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)