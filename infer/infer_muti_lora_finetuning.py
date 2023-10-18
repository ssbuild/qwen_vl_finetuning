# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper, global_args
from aigc_zoo.model_zoo.qwen_vl.llm_model import MyTransformer, QWenTokenizer, setup_model_profile, QWenConfig, \
    PetlArguments,PetlModel

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args)
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

    ckpt_dir = './best_ckpt/last'
    config = QWenConfig.from_pretrained(ckpt_dir)

    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    # new_num_tokens = config.vocab_size
    # if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
    #     config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,
                             # new_num_tokens=new_num_tokens,#扩充词
                             
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载多个lora权重
    pl_model.load_sft_weight(ckpt_dir,adapter_name="default")

    # 加载多个lora权重
    #pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")

    # 加载多个lora权重
    #pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")


    pl_model.eval().half().cuda()

    # backbone model replaced PetlModel
    lora_model: PetlModel = pl_model.backbone

    text_list = [
        "写一个诗歌，关于冬天",
        "晚上睡不着应该怎么办",
    ]
    generation_config = GenerationConfig(**{
        "chat_format": "chatml",
        "eos_token_id": tokenizer.eod_id,
        "max_new_tokens": 512,
        "pad_token_id": tokenizer.eod_id,
        #"stop_words_ids": [[tokenizer.eod_id]],
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
    })
    # 基准模型推理
    with lora_model.disable_adapter():
        for input in text_list:
            #lora_model 调用子对象方法
            response, history = lora_model.chat(tokenizer, input, history=[],generation_config=generation_config )
            print("input", input)
            print("response", response)

    lora_model.set_adapter(adapter_name='default')

    for input in text_list:
        # lora_model 调用子对象方法
        response, history = lora_model.chat(tokenizer, input, history=[],generation_config=generation_config )
        print("input", input)
        print("response", response)

