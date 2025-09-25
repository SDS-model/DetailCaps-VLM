from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *

image_file = ""

model_path = "./model-lora-finetune"
model_base = None

model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

args = type('Args', (), {
             "model": model,
             "tokenizer": tokenizer,
             "image_processor": image_processor,
             "context_len": context_len,
             "conv_mode": 'phi', 
             "image_file": image_file,
             "sep": ",",
             "temperature": 0,
             "top_p": None,
             "num_beams": 1,
             "max_new_tokens": 512
})()

#monitor = Monitor(args, model, llm_layers_index=31)
inference=eval_model(args)
print(inference)
#monitor.get_output(output_dir='results/')
