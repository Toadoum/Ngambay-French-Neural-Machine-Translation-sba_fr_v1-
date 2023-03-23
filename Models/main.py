from training import *
from training_arg import*
import transformers
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers import T5ForConditionalGeneration, AutoTokenizer
#from transformers import EarlyStoppingCallback, IntervalStrategy

args_dict = {
    "output_dir": "byt5_small_fr_sba_synth",
    "model_name_or_path": "google/byt5-small",
    "train_file": "/home/jupyter/Ngambay-French-Neural-Machine-Translation-sba_fr_v1-/Dataset/sba_fr_train_sy.json",
    "validation_file": "/home/jupyter/Ngambay-French-Neural-Machine-Translation-sba_fr_v1-/Dataset/sba_fr_val_sy.json",
    "test_file": "/home/jupyter/Ngambay-French-Neural-Machine-Translation-sba_fr_v1-/Dataset/sba_fr_test_sy.json",
    "source_lang": "fr",
    "target_lang": "sw",
    

    "max_source_length": "128",
    "max_target_length": "128",
    "num_train_epochs": "60",
    "per_device_train_batch_size": "5",
    "per_device_eval_batch_size": "5",
    "num_beams": "5",
    "save_steps": "10000",
    "seed": "65",
    

    "do_train": "True",
    "do_eval": "True",
    "do_predict":"True",
    "predict_with_generate": "True",
    "overwrite_output_dir": "True",
   

    # optional
    # for mt5
    # "source_prefix": "translate Frecnh to Wolof: ",
    # for m2m100
    #"forced_bos_token": "sw",
    # for mBART50
    #"forced_bos_token": "en_XX", # language code has _[country code]

}

model_args, data_args, training_args = get_args(args_dict)

start_training(model_args, data_args, training_args)

#Prediction function

model = T5ForConditionalGeneration.from_pretrained("/home/jupyter/Ngambay-French-Neural-Machine-Translation-sba_fr_v1-/Baseline/byt5_small_fr_sba_sy")
tokenizer = AutoTokenizer.from_pretrained("/home/jupyter/Ngambay-French-Neural-Machine-Translation-sba_fr_v1-/Baseline/byt5_small_fr_sba_sy")