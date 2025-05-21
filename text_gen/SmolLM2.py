from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
import eco2ai 
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

for i in range(30):
    eco2ai_tracker = eco2ai.Tracker(project_name=f"SmolLM2_text", file_name="text_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=f"SmolLM2_text", measure_power_secs=10,output_dir="text_gen", tracking_mode="process")
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [{"role": "user", "content": "Write 100 words of a novel simulating the style of Stephen King."}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=256, temperature=0.7, top_k = 50, top_p=0.95, do_sample=True)
    print(tokenizer.decode(outputs[0]))

    eco2ai_tracker.stop()
    CC_tracker.stop()