import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from codecarbon import EmissionsTracker
import eco2ai

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
for i in range(30):

    eco2ai_tracker = eco2ai.Tracker(project_name=f"Phi4_text", file_name="text_gen/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=f"Phi4_text", measure_power_secs=10, output_dir="text_gen", tracking_mode="process")
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [
        {"role": "system", "content": "You are an excellent chatbot who imitates writers to perfection."},
        {"role": "user", "content": "Write 100 words of a novel in the style of Stephen King"}
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
    print(output[0]['generated_text'])

    eco2ai_tracker.stop()
    CC_tracker.stop()
