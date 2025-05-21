import torch
from transformers import pipeline
from codecarbon import EmissionsTracker
import eco2ai
from ragas import SingleTurnSample
from ragas.metrics import BleuScore
import os
import csv


num_parameters = 2_610_000_000  # Number of parameters in the model
experiment_name = "Gemma2-2b_trans"
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

    eco2ai_tracker = eco2ai.Tracker(project_name=experiment_name, file_name="translation/eco2ai.csv")
    CC_tracker = EmissionsTracker(project_name=experiment_name, measure_power_secs=10, output_dir="translation", save_to_file=True)
    CC_tracker.start()
    eco2ai_tracker.start()

    messages = [
        {"role": "user", "content": "Translate this from English to Spanish:\English: In the suffocating gloom of the attic, Martha felt the weight of years pressing down on her. The floorboards creaked under her weight, each step a muffled symphony in the oppressive silence. She shone her flashlight, revealing cobiary lay open on the floor, its pages yellowed and brittle. She reached out, trembling, and opened it. The ink was still faint, but the words screamed of secrets best left buried. A chill ran down her spine as she read the first eer neck, but when she turned, there was nothing but darkness. The attic swallowed her, whispering promises of forgotten horrors.\nSpanish:"},
    ]

    outputs = pipe(messages, max_new_tokens=500, temperature=0.7, do_sample=True, top_k=50, top_p=0.95)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    print(assistant_response)

    eco2ai_tracker.stop()
    CC_tracker.stop()
    

    translation = assistant_response.split("**")[0]
    test_data = {
        "user_input": "Translate this from English to Spanish:\English: In the suffocating gloom of the attic, Martha felt the weight of years pressing down on her. The floorboards creaked under her weight, each step a muffled symphony in the oppressive silence. She shone her flashlight, revealing cobiary lay open on the floor, its pages yellowed and brittle. She reached out, trembling, and opened it. The ink was still faint, but the words screamed of secrets best left buried. A chill ran down her spine as she read the first eer neck, but when she turned, there was nothing but darkness. The attic swallowed her, whispering promises of forgotten horrors.\nSpanish:",
        "response": translation,
        "reference": "En la sofocante penumbra del ático, Martha sintió el peso de los años sobre ella. Las tablas del suelo se movían bajo su peso, cada paso una sinfonía amortiguada en el silencio opresivo. Alumbró con su linterna, revelando un libro abierto en el suelo, sus páginas amarillentas y quebradizas. Se acercó, temblando, y lo abrió. La tinta aún era tenue, pero las palabras gritaban secretos mejor guardados. Un escalofrío recorrió su columna vertebral mientras leyó la primera frase, pero al voltear, no había nada más que la oscuridad. El ático la engulló, susurrando promesas de horrores olvidados."
    }
    
    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    bleu = metric.single_turn_score(test_data)
    print(f"BLEU Score: {bleu}")  

    energy_consumed = CC_tracker.final_emissions_data.energy_consumed
    bleu_per_energy_cc = bleu / energy_consumed if energy_consumed > 0 else 0
    print(f"BLEU Score per Energy (CodeCarbon): {bleu_per_energy_cc}")
    
    energy_consumed_eco2ai = eco2ai_tracker.consumption()
    bleu_per_energy_eco2ai = bleu / energy_consumed_eco2ai if energy_consumed_eco2ai > 0 else 0
    print(f"BLEU Score per Energy (Eco2AI): {bleu_per_energy_eco2ai}")

    # Write to a new CSV file
    csv_file = "translation/metrics.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file is new
            writer.writerow(["Experiment Name", "Parameters", "BLEU Score", "BLEU/Energy (CodeCarbon)", "BLEU/Energy (Eco2AI)"])
        # Write the data
        writer.writerow([experiment_name, num_parameters, bleu, bleu_per_energy_cc, bleu_per_energy_eco2ai])
