# DetailCaps-VLM
A code repository for a visual language model suitable for generating captions for small objects in transmission line images.
## Installation and Requirements
1.Clone this repository and navigate to the folder
```bash
git clone https://github.com/SunDesheng-SDS/DetailCaps-VLM.git
cd DetailCaps-VLM
```
2.Install necessary packages

Our code is modified based on the TinyLLaVA framework. Please refer to [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) for this part.

## Get Started

1.Data Preparation

Our dataset mainly consists of two parts: pre-training and fine-tuning. The pre-training dataset uses [blip_laion_cc_sbu_558k](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main), and the fine-tuning dataset uses our self-built transmission line defect subtitle dataset. If you fine-tune your own model, please build it according to the format of the LLaVA dataset when building the fine-tuning dataset. If you need our transmission line dataset, please contact us privately.

2.Train

For training, you need to use the two files pretrain.sh and train_phi_lora.sh. The pretrain.sh file is used for pre-training the model, and the train_phi_lora.sh file is used for fine-tuning the model. Just replace the dataset path with your dataset path and download the pre-trained [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) and [phi-2](https://huggingface.co/microsoft/phi-2/tree/main) at the same time.

Pretrain
```bash
bash pretrain.sh
```

Finetune
```bash
bash train_phi_lora.sh
```

3.Evaluation

During the evaluation phase, we used classic image captioning evaluation metrics including BLEU-n, CIDEr, ROUGE-L, and METEOR. Our evaluation code is in the evaluate file. Two files are required for each indicator in the evaluation folder. One is the real label set for the images to be evaluated and saved as a JSON file. The other is the image folder that needs to be inferred by the model. The model will infer each image in the image folder to obtain captions and then calculate the corresponding indicator score with the label file.Please refer to the evaluate file for details.

Note:For reasoning on a single image, you can use our inference.py file.

4.Object Detection

For the object detection function, we use the [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) framework. If you need to use other frameworks, you only need to migrate the code library of the corresponding framework to our code library.

## Acknowledgement
We would like to thank Junlong Jia, Ying Hu, Xi Weng and others for building the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) framework.

We would like to thank Xizhou Zhu, Weijie Su, Lewei Lu and others for their work on [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).

## Community efforts
Our code repository is built on top of the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) project. Great work!

Our dataset format follows the [LLaVA](https://github.com/haotian-liu/LLaVA) instruction fine-tuning dataset format.Great work!



