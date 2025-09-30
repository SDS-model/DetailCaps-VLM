from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *

from rouge import Rouge
import json
score_ROUGE_sum = 0.0

model_path = "./model-lora-finetune"
model_base = None
model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

with open('./data/val/val.json', 'r') as dataset_Validation_file:
    dataset_Validation_data = json.load(dataset_Validation_file)
    for i in range(len(dataset_Validation_data)):
        # 定义参考摘要和候选摘要
        reference = dataset_Validation_data[i]['conversations'][1]['value']
        image_list = dataset_Validation_data[i]['image']
        image_file = "./data/val/data/"+image_list
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
        candidate = eval_model(args)           #对应模型输出

        # 创建Rouge对象
        rouge = Rouge()

        # 计算ROUGE分数
        scores = rouge.get_scores(candidate, reference)

        # 提取ROUGE-L分数
        rouge_l_score = scores[0]['rouge-l']
        score_ROUGE_sum += rouge_l_score['f']
        if i%10==0:
            print(score_ROUGE_sum)
print(f"ROUGE-L F1: {score_ROUGE_sum}")