# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser, GenerationConfig
from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.qwen_vl.llm_model import MyTransformer,QWenTokenizer,setup_model_profile, QWenConfig,PetlArguments


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)
    setup_model_profile()
    dataHelper = NN_DataHelper(model_args)
    tokenizer: QWenTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=QWenTokenizer, config_class_name=QWenConfig)

    weight_dir = '../scripts/best_ckpt'
    lora_weight_dir = os.path.join(weight_dir, "last")

    config = QWenConfig.from_pretrained(weight_dir)
    lora_args = PetlArguments.from_pretrained(lora_weight_dir)

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
    # 加载lora权重
    pl_model.load_sft_weight(lora_weight_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(lora_weight_dir,'pytorch_model_merge.bin'),merge_lora_weight=True)

    else:
        model = pl_model.get_llm_model()

        text_list = [
            "写一个诗歌，关于冬天",
            "晚上睡不着应该怎么办",
        ]

        model.generation_config = GenerationConfig(**{
            "chat_format": "chatml",
            "eos_token_id": tokenizer.eod_id,
            "max_new_tokens": 512,
            "pad_token_id": tokenizer.eod_id,
            #"stop_words_ids": [[tokenizer.eod_id]],
            "do_sample": True,
            "top_k": 0,
            "top_p": 0.8,
        })
        for input in text_list:
            response, history = model.chat(tokenizer, input, history=[],)
            print("input", input)
            print("response", response)



