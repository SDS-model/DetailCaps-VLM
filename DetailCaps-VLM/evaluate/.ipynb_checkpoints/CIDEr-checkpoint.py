from tinyllava.eval.run_tiny_llava import eval_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from tinyllava_visualizer.tinyllava_visualizer import *


#from pycocoevalcap.cider.cider import Cider
import json

import math
from collections import defaultdict
import numpy as np
def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    return precook(test, n, True)


class CiderScorer(object):
    def __init__(self, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        self.crefs.append(cook_refs(refs))
        self.ctest.append(cook_test(test))

    def compute_doc_freq(self):
        document_frequency = defaultdict(float)
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                document_frequency[ngram] += 1
        return document_frequency

    def compute_cider(self):
        def counts2vec(cnts):
            vec = defaultdict(float)
            length = 0
            for ngram, term_freq in cnts.items():
                vec[ngram] = term_freq
                length += term_freq ** 2
            length = math.sqrt(length)
            return vec, length

        def sim(vec_hyp, vec_ref, length_hyp, length_ref, doc_freq, document_count):
            delta = defaultdict(float)
            for ngram, term_freq in vec_hyp.items():
                delta[ngram] = term_freq
            for ngram, term_freq in vec_ref.items():
                delta[ngram] -= term_freq
            norm = sum([x ** 2 for x in delta.values() if x != 0])
            return np.exp(-norm / (2 * self.sigma ** 2))

        document_count = len(self.crefs)
        document_frequency = self.compute_doc_freq()
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            score = np.array([0.0] * self.n)
            vec, norm = counts2vec(test)
            for ref in refs:
                ref_vec, ref_norm = counts2vec(ref)
                for n in range(self.n):
                    subvec = {ngram: term_freq for ngram, term_freq in vec.items() if len(ngram) == n + 1}
                    subref_vec = {ngram: term_freq for ngram, term_freq in ref_vec.items() if len(ngram) == n + 1}
                    subnorm = math.sqrt(sum([x ** 2 for x in subvec.values()]))
                    subref_norm = math.sqrt(sum([x ** 2 for x in subref_vec.values()]))
                    score[n] += sim(subvec, subref_vec, subnorm, subref_norm, document_frequency, document_count)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(score), np.array(score)


def calculate_cider_score(test_sentences, reference_sentences):
    scorer = CiderScorer()
    for test, refs in zip(test_sentences, reference_sentences):
        scorer.cook_append(test, refs)
    score, scores = scorer.compute_score()
    return score
# 模拟参考描述和生成描述
# 这里假设参考描述和生成描述是按照图像ID组织的
# 图像ID可以是任意字符串，只要能唯一标识图像即可
# 参考描述是一个字典，键是图像ID，值是该图像对应的多个参考描述列表
scider_sum = 0
model_path = "./model-lora-finetune"
model_base = None
model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
references = {}
candidates = {}
with open('./data/val/val.json', 'r') as dataset_Validation_file:
    dataset_Validation_data = json.load(dataset_Validation_file)
    for i in range(len(dataset_Validation_data)):
        references[dataset_Validation_data[i]['id']] = [dataset_Validation_data[i]['conversations'][1]['value']]
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
        candidates[dataset_Validation_data[i]['id']] = [eval_model(args)]#模型的输出
        cider_score = calculate_cider_score(candidates[dataset_Validation_data[i]['id']],dataset_Validation_data[i]['conversations'][1]['value'])
        #print(candidates[dataset_Validation_data[i]['id']][0],dataset_Validation_data[i]['conversations'][1]['value'])
        print(cider_score)
        scider_sum+=cider_score
    #ref = {}
    #can = {}
    #for img_id, ref_list in references.items():
    #    ref[img_id] = references[img_id]
    #for img_id, cand in candidates.items():
    #    can[img_id] = candidates[img_id]
    #scorer = Cider()
    #score, scores = scorer.compute_score(ref, can)
    print(f"CIDEr score: {scider_sum}")  #这里算的就是所有图片的平均值所以没有累加