import os

from collections import OrderedDict

import torch
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.tuners.lora import LoraLayer

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils.train_utils import *
from ..utils import log
from ..model import TinyLlavaConfig, TinyLlavaForConditionalGeneration


@register_training_recipe('lora')
class LoRATrainingRecipe(BaseTrainingRecipe):
    def __init__(self, training_arguments):
        super().__init__(training_arguments)
        self.training_arguments = training_arguments
        self.lora_skip_module = ['connector', 'vision_tower', 'language_model']
        
        
    def training_model_converse(self, model):
        if self.training_arguments.tune_type_connector == 'lora':
            self.lora_skip_module.remove('connector')
        if self.training_arguments.tune_type_llm == 'lora':
            self.lora_skip_module.remove('language_model')
        if self.training_arguments.tune_type_vision_tower == 'lora':
            self.lora_skip_module.remove('vision_tower')
        lora_config = LoraConfig(
            r=self.training_arguments.lora_r,
            lora_alpha=self.training_arguments.lora_alpha,
            target_modules=find_all_linear_names(model, self.lora_skip_module),
            lora_dropout=self.training_arguments.lora_dropout,
            bias=self.training_arguments.lora_bias,
            task_type="CAUSAL_LM",
        )
        if self.training_arguments.bits == 16:
            if self.training_arguments.bf16:
                model.to(torch.bfloat16)
            if self.training_arguments.fp16:
                model.to(torch.float16)
        if not hasattr(model, 'peft_config') or model.peft_config is None:
            log("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)  
        return model
        
        
    def save(self, model, trainer):
        model.config.use_cache = True
        #save tokenizer       
        model.tokenizer.save_pretrained(self.training_arguments.output_dir)
        #save entire model config
        model.config.save_pretrained(self.training_arguments.output_dir, from_pt=True)
        #save trainer
        trainer.save_state()

        if 'finetune' in self.training_arguments.output_dir and self.training_arguments.pretrained_model_path is not None: # for finetune stage
            if trainer.deepspeed:
                torch.cuda.synchronize()
            trainer.save_model(self.training_arguments.output_dir,save_safetensors=True)
            return
        
        #save language model base params
        language_model_state_dict = get_peft_state_non_lora_maybe_zero_3(model.language_model.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            language_model_output_dir = os.path.join(self.training_arguments.output_dir, 'language_model')
            os.makedirs(language_model_output_dir, exist_ok=True)
            language_model_output_path = os.path.join(self.training_arguments.output_dir, 'language_model/pytorch_model.bin')
            torch.save(language_model_state_dict, language_model_output_path)
            model.config.text_config.save_pretrained(language_model_output_dir, from_pt=True)
        #save vision tower base params
        vision_tower_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower._vision_tower.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            vision_tower_output_dir = os.path.join(self.training_arguments.output_dir, 'vision_tower')
            os.makedirs(vision_tower_output_dir, exist_ok=True)
            vision_tower_output_path = os.path.join(self.training_arguments.output_dir, 'vision_tower/pytorch_model.bin')
            torch.save(vision_tower_state_dict, vision_tower_output_path)
            model.config.vision_config.save_pretrained(vision_tower_output_dir, from_pt=True)
        #save connector base params
        connector_state_dict = get_peft_state_non_lora_maybe_zero_3(model.connector.named_parameters(),  False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            connector_output_dir = os.path.join(self.training_arguments.output_dir, 'connector')
            os.makedirs(connector_output_dir, exist_ok=True)
            connector_output_path = os.path.join(self.training_arguments.output_dir, 'connector/pytorch_model.bin')
            torch.save(connector_state_dict, connector_output_path)        
        #save EnhancedCMN-model
        EnhancedCMN_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower.EnhancedCMN.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            EnhancedCMN_output_dir = os.path.join(self.training_arguments.output_dir, 'EnhancedCMN')
            os.makedirs(EnhancedCMN_output_dir, exist_ok=True)
            EnhancedCMN_output_path = os.path.join(self.training_arguments.output_dir, 'EnhancedCMN/pytorch_model.bin')
            torch.save(EnhancedCMN_state_dict, EnhancedCMN_output_path)
        #save GlobalFeatureEnhancement-model
        GlobalFeatureEnhancement_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower.GlobalFeatureEnhancement.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            GlobalFeatureEnhancement_output_dir = os.path.join(self.training_arguments.output_dir, 'GlobalFeatureEnhancement')
            os.makedirs(GlobalFeatureEnhancement_output_dir, exist_ok=True)
            GlobalFeatureEnhancement_output_path = os.path.join(self.training_arguments.output_dir, 'GlobalFeatureEnhancement/pytorch_model.bin')
            torch.save(GlobalFeatureEnhancement_state_dict, GlobalFeatureEnhancement_output_path) 
        #save MultiViewFeatureFusion-model
        MultiViewFeatureFusion_state_dict = get_peft_state_non_lora_maybe_zero_3(model.vision_tower.MultiViewFeatureFusion.named_parameters(), False)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            MultiViewFeatureFusion_output_dir = os.path.join(self.training_arguments.output_dir, 'MultiViewFeatureFusion')
            os.makedirs(MultiViewFeatureFusion_output_dir, exist_ok=True)
            MultiViewFeatureFusion_output_path = os.path.join(self.training_arguments.output_dir, 'MultiViewFeatureFusion/pytorch_model.bin')
            torch.save(MultiViewFeatureFusion_state_dict, MultiViewFeatureFusion_output_path) 
        # save lora params
        lora_state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), self.training_arguments.lora_bias
        )
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            model.save_pretrained(self.training_arguments.output_dir, state_dict=lora_state_dict)
        

