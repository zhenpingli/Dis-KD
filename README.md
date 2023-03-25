"""
# Discriminator enhanced Knowledge Distillation Networks


To run the code please follow:

python main.py \
    --student_type gpt2 \
    --student_config /home/Gan-Distill/config2.json \
    --teacher_type gpt2 \
    --teacher_name gpt2 \
    --alpha_ce 0  --alpha_cos 0 --alpha_clm 1 \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.gpt2.pickle \
    --token_counts data/token_counts.gpt2.pickle \
    --force 

"""
