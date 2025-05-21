import torch
from transformers import pipeline
from ragas import SingleTurnSample
from ragas.metrics import StringPresence
from codecarbon import EmissionsTracker
import eco2ai
import csv
import os

experiment_name = "Gemma2-2b_code"
num_parameters = 2_610_000_000 
access_token = ""  # Replace with your Hugging Face access token

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
    token= access_token,
    trust_remote_code=True,
)

for i in range(30):
    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="code_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="code_gen", save_to_file=True, tracking_mode="process")
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [
        {"role": "user", "content": "Write a template for the inference of a model from Hugging Face using Transformers and adjusting the maximum new tokens to 500."},
    ]

    outputs = pipe(messages, max_new_tokens=500, temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    print(assistant_response)
   
    eco2ai_tracker.stop()
    CC_tracker.stop()
    
    test_data = {
        "response": assistant_response,
        "reference": "max_new_tokens",
    }
    scorer = StringPresence()
    test_data = SingleTurnSample(**test_data)
    presence = scorer.single_turn_score(test_data)
    print(f"String Pressence: {presence}")  

    # Write to a new CSV file
    csv_file = "code_gen/metrics.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file is new
            writer.writerow(["Experiment Name", "Parameters", "String Presence"])
        # Write the data
        writer.writerow([experiment_name, num_parameters, presence])

