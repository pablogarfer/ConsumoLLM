import torch
from transformers import pipeline
from codecarbon import EmissionsTracker
import eco2ai

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

for i in range(30):

    eco2ai_tracker = eco2ai.Tracker(project_name=f"TinyLLama_text", file_name="text_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=f"TinyLLama_text", measure_power_secs=10, output_dir="text_gen", tracking_mode="process")
    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    CC_tracker.start()
    eco2ai_tracker.start()
    messages = [
        {
            "role": "system",
            "content": "You are an excellent chatbot who imitates writers to perfection.",
        },
        {"role": "user", "content": "Write 100 words of a novel simulating the style of Stephen King."},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    CC_tracker.stop()
    eco2ai_tracker.stop()

