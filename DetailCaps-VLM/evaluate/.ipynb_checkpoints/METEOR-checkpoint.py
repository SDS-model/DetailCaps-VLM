from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *

import nltk
# 下载必要的nltk数据集
nltk.download('wordnet')
#nltk.download('punkt')
from nltk.translate.meteor_score import meteor_score
import json

score_METEOR_sum = 0.0
reference_list = []
candidate_list = []
i = 0

model_path = "./model-lora-finetune"
model_base = None
model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

# 从JSON文件中读取数据
with open('./data/val/val.json', 'r') as dataset_Validation_file:
    dataset_Validation_data = json.load(dataset_Validation_file)
    for i in range(len(dataset_Validation_data)):
        # 假设参考译文在 'conversations' 列表的第二个元素的 'value' 字段
        reference_list.append(dataset_Validation_data[i]['conversations'][1]['value'])
        # 假设候选译文在 'conversations' 列表的第一个元素的 'value' 字段，根据实际情况修改
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
        candidate_list.append(eval_model(args))

# 确保参考译文列表和候选译文列表长度一致
if len(reference_list) != len(candidate_list):
    raise ValueError("参考译文列表和候选译文列表长度不一致")

# 计算每个样本的METEOR分数并累加
for reference, candidate in zip(reference_list, candidate_list):
    # 将参考译文拆分为单词列表
    references = [ref.split() for ref in [reference]]
    # 将候选译文拆分为单词列表
    candidate_tokens = candidate.split()

    # 计算METEOR分数
    score = meteor_score(references, candidate_tokens)
    score_METEOR_sum += score
    if i%10==0:
        print(score_METEOR_sum)
    i=i+1
print(score_METEOR_sum)