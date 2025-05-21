from transformers import AutoModelForCausalLM, AutoTokenizer
from ragas import SingleTurnSample
from ragas.metrics import StringPresence
from codecarbon import EmissionsTracker
import eco2ai 
import csv
import os

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
experiment_name = "qwen2.5-coder"
num_parameters = 3_090_000_000 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for i in range(30):
     
    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="code_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="code_gen", save_to_file=True, tracking_mode="process")
    CC_tracker.start()
    eco2ai_tracker.start()
    
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Write a template for the inference of a model from Hugging Face using Transformers and adjusting the maximum new tokens to 500."},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=500, temperature=0.7, 
        top_k = 50, 
        top_p=0.95, 
        do_sample=True
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(output)

    eco2ai_tracker.stop()
    CC_tracker.stop()
    
    
    test_data = {
        "response": output,
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

