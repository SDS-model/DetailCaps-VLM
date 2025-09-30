from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *


import jieba
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import json
score1_sum = 0.0
score2_sum = 0.0
score3_sum = 0.0
score4_sum = 0.0
model_path = "./model-lora-finetune"
model_base = None

model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
sf = SmoothingFunction()
with open('./data_damage/val/val.json', 'r') as dataset_Validation_file:
    dataset_Validation_data = json.load(dataset_Validation_file)
    for i in range(len(dataset_Validation_data)):
        target = dataset_Validation_data[i]['conversations'][1]['value']  # target
        #print(target)
        image_list = dataset_Validation_data[i]['image']
        image_file = "./data_damage/val/damage/"+image_list
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
        inference = eval_model(args)   # inference,这里后面改成模型的输出就ok了
        #print(inference)
        # 分词
        target_fenci = ' '.join(jieba.cut(target))
        inference_fenci = ' '.join(jieba.cut(inference))
        reference = []  # 给定标准译文
        candidate = []  # 神经网络生成的句子
        # 计算BLEU
        reference.append(target_fenci.split())
        candidate = (inference_fenci.split())
        score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0),smoothing_function=sf.method1)
        score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0),smoothing_function=sf.method1)
        score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0),smoothing_function=sf.method1)
        score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1),smoothing_function=sf.method1)
        reference.clear()
        score1_sum += score1
        score2_sum += score2
        score3_sum += score3
        score4_sum += score4
        if i%10 == 0:
            print(score1_sum,score2_sum,score3_sum,score4_sum)
    print('Cumulate 1-gram :%f' \
           % score1_sum)
    print('Cumulate 2-gram :%f' \
           % score2_sum)
    print('Cumulate 3-gram :%f' \
           % score3_sum)
    print('Cumulate 4-gram :%f' \
           % score4_sum)

