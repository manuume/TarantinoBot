# 🎬 Tarantino Scene Generator: A Fine-Tuned Creative LLM




**Live Demo:** [(https://huggingface.co/spaces/manohar3181/taratino-bot)](https://huggingface.co/spaces/manohar3181/taratino-bot)

**Fine-Tuned Model:** [manohar3181/Tarantino-Scene-Generator-v1](https://huggingface.co/manohar3181/TarintinoStyle-scene-generator-v1)

**Training Dataset:** [manohar3181/TarantinoBot-Finetuning-Dataset](https://huggingface.co/datasets/manohar3181/TarantinoBot-Finetuning-Dataset)


 a specialized model fine-tuned to write like Quentin Tarantino. The goal was to move beyond generic scripts and capture the director's unique signature: the witty, pop-culture-laden dialogue, the slow-burn tension, the specific screenplay formatting, and the sudden tonal shifts.

The result is a creative tool that takes a simple idea and generates a complete, formatted movie scene in an authentic Tarantino-esque style.



## 📊 2. The Data Strategy

The success of this project hinged on a robust data preparation strategy. The process transformed raw, unstructured movie scripts into a high-quality, structured dataset suitable for fine-tuning.

### Phase 1: Sourcing & Structuring

**Data Collection:** Sourced plain-text versions of seven iconic screenplays to ensure a consistent base format. The corpus included a diverse mix of Tarantino's work to capture his stylistic range.

**Parsing Pipeline:** Developed a custom Python parser using regular expressions and heuristic rules to deconstruct the raw text, segmenting it into individual scenes and classifying each line as a heading, action, or dialogue.

**Intermediate Format:** The output was a master structured.json file containing 416 parsed scenes, creating a clean and reusable blueprint of the entire script corpus.

### Phase 2: AI-Assisted Instruction Generation

**The Challenge:** Manually writing 416 high-quality, creative prompts was not feasible.

**The Solution:** Implemented an AI-assisted data generation pipeline. For each scene, a powerful LLM (Llama 3 via Groq) was given a sophisticated "meta-prompt," instructing it to act as an expert screenwriter and generate a high-level creative brief that captured the scene's core conflict and tonal shifts.

**Final Dataset:** The final output was a finetuning.json file containing 416 (instruction, response) pairs, which was then uploaded to the Hugging Face Hub.

## 🧠 3. The Base Model

The choice of the base model was critical, especially given the constraints of free cloud GPU environments.

**Model:** mistralai/Mistral-7B-Instruct-v0.2

**Reasoning:**

- **Instruction-Tuned:** An Instruct model already knows how to follow commands, which was crucial for getting the script format right.
- **Performance:** Mistral 7B is renowned for its high performance and quality, making it ideal for creative tasks.
- **Feasibility:** At 7B parameters, it's at the upper limit of what can be fine-tuned on free-tier cloud GPUs using QLoRA.

## 📓 4. The Notebooks

This repository contains the notebooks that document the entire project lifecycle.

- **1_Dataset_creation.ipynb:** The complete data pipeline, from parsing raw scripts to generating the final training dataset with AI-assisted instructions.
- **2_model_trainingv2.ipynb:** The script used to fine-tune the Mistral-7B-Instruct model using QLoRA and optimized TrainingArguments.
- **3_Testver2.ipynb:** A notebook for qualitative evaluation, loading the fine-tuned model from a specific checkpoint to run inference.
- **3_app.py:** app creation using gradio.
- **3_hg_push.ipynb:** pushing to hugging face.
- 

## 🚀 5. How to Run the Model

The final, merged model is publicly available on the Hugging Face Hub. You can easily use it for inference with the transformers library.

```python
from transformers import pipeline
import torch

# Make sure to replace with your final model name
MODEL_NAME = "manohar3181/Tarantino-Scene-Generator-v1"

generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Write a scene where two veteran bank robbers in a getaway car argue about the best type of donut to eat after a heist."
# Use the official Mistral chat template
formatted_prompt = f"<s>[INST] {prompt} [/INST]"

outputs = generator(formatted_prompt, max_new_tokens=400)
print(outputs[0]['generated_text'].split("[/INST]")[1].strip())
```
