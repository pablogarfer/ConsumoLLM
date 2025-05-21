from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
import eco2ai 
from ragas import SingleTurnSample
from ragas.metrics import StringPresence
import csv
import os

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
num_parameters = 1_710_000_000  
experiment_name = "SmolLM2_code"


device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

for i in range(30):
    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="code_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="code_gen", save_to_file=True, tracking_mode="process")
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [{"role": "user", "content": "Write a template for the inference of a model from Hugging Face using Transformers and adjusting the maximum new tokens to 500."}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=500, temperature=0.7, top_k = 50, top_p=0.95, do_sample=True)
    assistant_response = tokenizer.decode(outputs[0])
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