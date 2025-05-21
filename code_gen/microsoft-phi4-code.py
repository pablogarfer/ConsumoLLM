import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from codecarbon import EmissionsTracker
import eco2ai
from ragas import SingleTurnSample
from ragas.metrics import StringPresence
import csv
import os

model_path = "microsoft/Phi-4-mini-instruct"
num_parameters = 3_840_000_000  
experiment_name = "Phi4_code"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
for i in range(30):

    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="code_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="code_gen", save_to_file=True, tracking_mode="process")   
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [
        {"role": "system", "content": "You are an excellent chatbot who writes efficient code."},
        {"role": "user", "content": "Write a template for the inference of a model from Hugging Face using Transformers and adjusting the maximum new tokens to 500."}
        #{"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        #{"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        #{"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]
 
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    generation_args = {
        "max_new_tokens": 256,
        "return_full_text": False,
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
    }
    
    output = pipe(messages, **generation_args)
    assistant_response = output[0]["generated_text"]
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
