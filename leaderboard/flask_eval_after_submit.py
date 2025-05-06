from flask import Flask, request, jsonify

app = flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    print(f"🚀 Evaluation triggered for model: {data.get('model')}")

    return jsonify({"status": "evaluation started", "model": data.get("model")})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)