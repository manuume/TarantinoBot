import gradio as gr
from transformers import pipeline
import time
import os
import torch

MODEL_NAME = "manohar3181/TarintinoStyle-scene-generator-v1"
HF_TOKEN = os.environ.get("HF_TOKEN")
start_time=time.time()
generator = pipeline("text-generation", model=MODEL_NAME,token=HF_TOKEN,torch_dtype=torch.bfloat16,device_map="auto")
end_time = time.time()



def generate_scene(prompt):
    if generator is None:
        return "Hold on, hold on, somethin' ain't right with the model load. Check the logs, dig?"
        
    print(f"Received prompt: {prompt}")
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    outputs = generator(
        formatted_prompt,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.75,
        top_k=50,
        top_p=0.95
    )
    
    generated_text = outputs[0]['generated_text'].split("[/INST]")[1].strip()
    return generated_text

css = """
.dark h1, .dark .prose { color: #FFFFFF; }
"""

with gr.Blocks(theme=gr.themes.Soft(),css=css) as demo:
    gr.Markdown("#Tarantino style movie Scene Generator")
    gr.Markdown("Fan of Tarantino's writing, huh? Ever wanna cook up a scene that sounds EXACTLY like somethin' outta *Pulp Fiction* or *Reservoir Dogs*? This here model will generate a movie scene according to your prompt in **Tarantinoesque** style. Go on, tell me what you got.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Alright, spit it out. What kinda scene you got cookin'?",
                lines=3,
                placeholder="e.g., Two hitmen discuss the merits of different breakfast cereals while on a stakeout."
            )
            
            generate_button = gr.Button("GENERATE")
            
            output_text = gr.Textbox(label="Here's the Scene:", lines=15)
        
        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    ["Two low-level thugs argue about the proper way to make a cup of coffee before a big job."],
                    ["A charismatic but deadly femme fatale sips a milkshake while negotiating a dangerous deal."],
                    ["A nervous rookie tries to explain a disastrous situation to his cool and collected boss."],
                ],
                inputs=prompt_input,
                label="Click an example to try it out"
            )

    generate_button.click(
        fn=generate_scene,
        inputs=prompt_input,
        outputs=output_text
    )
demo.launch(share=True)