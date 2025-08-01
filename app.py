from flask import Flask, request, render_template
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning import APIClient

app = Flask(__name__)

# IBM Watsonx credentials
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "JZ4P9-fJmcGv10Ge6OEFyJUNL_ZsUkmZrCv2IfLAoWU6"
}
project_id = "efa124a5-42ad-4202-b6d9-b184649096e6"

# Initialize Watsonx client
client = APIClient(wml_credentials)
client.set.default_project(project_id)

# Load model
granite_model = Model(
    model_id="ibm/granite-3-8b-instruct",
    credentials=wml_credentials,
    project_id=project_id
)

def build_prompt(idea):
    return f"""
You are a Startup Assistant Agent.

Given the idea: "{idea}", generate a structured startup business blueprint including:

1. Business Model Canvas
2. Estimated Budget
3. Market Research
4. Competitor Analysis
5. Go-to-Market Strategy
6. Potential Funding Options
7. Relevant Indian Government Schemes
8. Legal and Regulatory Requirements

Output each section clearly.
"""


def generate_blueprint(startup_idea):
    prompt = build_prompt(startup_idea)
    parameters = {
        "max_new_tokens": 8192,
        "decoding_method": "greedy"
    }
    response = granite_model.generate(prompt=prompt, params=parameters)
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        idea = request.form["idea"]

        response = generate_blueprint(idea)
        result = response.get("results", "No response generated.")
        result= result[0].get("generated_text", "No Response")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
