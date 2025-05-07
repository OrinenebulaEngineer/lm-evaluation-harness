import requests
from flask import Flask, request, jsonify
from evaluate_by_harness import evaluate  # This is your local module
from pathlib import Path

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def trigger_evaluation():
    # ✅ Fix: Typo in `request.json` (was `jsony`)
    data = request.json
    model = data.get('model')

    print(f"🚀 Evaluation triggered for model: {model}")

    # ✅ Call your local evaluate function
    eval_response = evaluate(model)

    # ✅ Optional: Forward the evaluation result to another server if needed
    payload = {"model": model, "eval_result": eval_response}
    try:
        forward_response = requests.post("http://localhost:5001/evaluate", json=payload)
        forward_status = forward_response.status_code
    except Exception as e:
        forward_status = f"Failed to forward: {e}"

    return jsonify({
        "status": "evaluation started",
        "model": model,
        "eval_result": eval_response,
        "forward_status": forward_status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
