import torch
from transformers import pipeline
from codecarbon import EmissionsTracker
import eco2ai
from ragas import SingleTurnSample
from ragas.metrics import StringPresence
import csv
import os

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32, device_map="auto")
num_parameters = 1_100_000_000  # Example: 1.1 billion parameters for TinyLlama
experiment_name = "TinyLLama_code"

for i in range(30):

    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="code_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="code_gen", save_to_file=True, tracking_mode="process")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    CC_tracker.start()
    eco2ai_tracker.start()
    messages = [
        {
            "role": "system",
            "content": "You are an excellent chatbot who writes efficient code.",
        },
        {"role": "user", "content": "Write a template for the inference of a model from Hugging Face using Transformers and adjusting the maximum new tokens to 500."},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    assistant_response = outputs[0]["generated_text"]
    print(assistant_response)
    CC_tracker.stop()
    eco2ai_tracker.stop()

    test_data = {
        "response": assistant_response,
        "reference": "max_new_tokens",
    }
    scorer = StringPresence()
    test_data = SingleTurnSample(**test_data)
    print(scorer.single_turn_score(test_data))

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