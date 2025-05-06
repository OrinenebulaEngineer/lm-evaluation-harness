from flask import Flask, request, jsonify
from evaluate_by_harness import evaluate

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    model = data.get('model')
    print(f"🚀 Evaluation triggered for model: {model}")

    # Call the evaluate_tasks function from the evaluate module
    eval_response = evaluate(model )
    return jsonify({"status": "evaluation started", "model": model, "eval_result" : eval_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)