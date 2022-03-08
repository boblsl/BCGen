import texar as tx

dcoder_config = {
    'dim': 768,
    'num_blocks': 12,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 768
    },
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}

loss_label_confidence = 0.9

random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768

opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
    }
}

lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 10000,
}

bos_token_id = 101
eos_token_id = 102

model_dir = "./models_wash"
run_mode = "train_and_evaluate"

# the batch size of train, valid and test set
batch_size = 32
eval_batch_size = 32
test_batch_size = 32
# batch_size = 16
# eval_batch_size = 16
# test_batch_size = 16

max_train_steps = 100000

display_steps = 10
checkpoint_steps = 2000
eval_steps = 50000
test_steps = 5000

max_decoding_length = 400

max_seq_length_src = 384
max_seq_length_cfg = 384
max_seq_length_tgt = 64

epochs = 10

is_distributed = False

data_dir = "datawash/data/"

train_out_file = "datawash/data/train.tf_record"
eval_out_file = "datawash/data/eval.tf_record"
test_out_file = "datawash/data/test.tf_record"

bert_pretrain_dir="./bert_uncased_model"

train_story = "datawash/data/train_story.txt"
train_summ = "datawash/data/train_summ.txt"
train_cfg = "datawash/data/train_cfg.txt"

eval_story = "datawash/data/eval_story.txt"
eval_summ = "datawash/data/eval_summ.txt"
eval_cfg = "datawash/data/eval_cfg.txt"

test_story = "datawash/data/test_story.txt"
test_cfg = "datawash/data/test_cfg.txt"
test_summ = "datawash/data/test_summ.txt"


bert_pretrain_dir = "./pretrain_model"
