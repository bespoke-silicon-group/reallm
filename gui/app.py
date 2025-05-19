from flask import Flask, render_template, request
import math
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.base.model import deepseekv2, llama70b


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    mermaid_code = None

    if request.method == "POST":
        model = request.form["model"]
        try:
            if model == "llama70b":
                model_class = llama70b
            # elif model == "deepseekv2":
            #     model_class = deepseekv2
            else:
                raise ValueError("Invalid model selected.")

            devices = int(request.form["devices"])
            D = int(request.form.get("data_parallel") or 1)
            T = int(request.form.get("tensor_parallel") or 1)
            P = int(request.form.get("pipeline_parallel") or 1)
            C = int(request.form.get("context_parallel") or 1)
            E = int(request.form.get("expert_parallel") or 1 if model == "deepseekv2" else 1)

            if any(x <= 0 for x in [devices, D, T, P, C, E]):
                raise ValueError("All values must be positive integers.")

            product = D * T * P * C * E
            if product != devices:
                raise ValueError(f"Product of parallelisms = {product}, which must equal number of devices = {devices}.")
            
            prefill_len = int(request.form.get("prefill_length") or 0)

            num_decode_tasks = int(request.form.get("num_decode_tasks") or 0)
            decode_task_context_length = int(request.form.get("decode_task_context_length") or 1)
            decode_tasks = num_decode_tasks * [decode_task_context_length]

            mermaid_code, hover_infos = model_class.generate_layer_mermaid(False, prefill_len, decode_tasks, (E, T, P, C))


            return render_template("index.html", mermaid_code=mermaid_code, hover_infos=hover_infos, error=None)

        except ValueError as ve:
            error = str(ve)

    return render_template("index.html", error=error, mermaid_code=mermaid_code, hover_infos=None)

if __name__ == "__main__":
    app.run(debug=True)
