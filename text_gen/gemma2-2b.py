import torch
from transformers import pipeline
from codecarbon import EmissionsTracker
import eco2ai


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
    CC_tracker = EmissionsTracker(project_name=f"Gemma2-2B-text", output_dir="text_gen", measure_power_secs=10, tracking_mode="process")
    eco2ai_tracker = eco2ai.Tracker(project_name=f"Gemma2-2B-text", file_name="text_gen/eco2ai.csv")
    
    CC_tracker.start()
    eco2ai_tracker.start()
    messages = [
        {"role": "user", "content": "Write 100 words of a novel imitating the style of Stephen King."},
    ]

    outputs = pipe(messages, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()  # Extract the assistant's response from the output
    print(assistant_response)
    
    CC_tracker.stop()
    eco2ai_tracker.stop()
