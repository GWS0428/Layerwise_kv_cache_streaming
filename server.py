from flask import Flask, request, send_file, abort
import os

app = Flask(__name__)

ENCODED_DIR = "/encoded"
num_layers = 32

@app.route("/get_kv_cache", methods=["GET"])
def get_kv_cache():
    doc_id = request.args.get("doc_id")

    if doc_id is None:
        abort(400, description="doc_id parameter is required.")

    for l in range(num_layers):
        file_path = os.path.join(ENCODED_DIR, f"{doc_id}_layer_{l}.pkl")

        if not os.path.exists(file_path):
            abort(404, description="File not found")

        return send_file(file_path, mimetype='application/octet-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
