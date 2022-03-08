import sys

if not 'texar_repo' in sys.path:
    sys.path += ['texar_repo']
import tensorflow as tf
import texar as tx
import numpy as np
from config import *
from BCGen import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if not os.path.exists('./output'):
    os.mkdir('./output')
def _eval_epoch(sess, epoch, mode):
    cfg, diff, references, hypotheses = [], [], [], []

    if mode == 'test':
        iterator.restart_dataset(sess, 'test')
        bsize = test_batch_size
        fetches = {
            'inferred_ids': inferred_ids,
        }
        bno = 0

    with open('./output/refs_wash.txt', 'w') as f1, open('./output/hyps_wash.txt', 'w') as f2,\
            open('./output/srcs_wash.txt', 'w') as f3, open('./output/cfgs_wash.txt', 'w') as f4:
        while True:

            # print("Temp",temp)
            try:
                print("Batch", bno)
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                op = sess.run([batch], feed_dict)

                feed_dict = {
                    src_input_ids: op[0]['src_input_ids'],
                    src_segment_ids: op[0]['src_segment_ids'],
                    cfg_input_ids: op[0]['cfg_input_ids'],
                    cfg_segment_ids: op[0]['cfg_segment_ids'],
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT
                }
                fetches_ = sess.run(fetches, feed_dict=feed_dict)
                labels = op[0]['tgt_labels']
                diffs = op[0]['src_input_ids']
                cfgs = op[0]['cfg_input_ids']
                hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
                references.extend(r.tolist() for r in labels)
                diff.extend(d.tolist() for d in diffs)
                cfg.extend(d.tolist() for d in cfgs)

                bno = bno + 1

            except tf.errors.OutOfRangeError:
                break

        hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
        references = utils.list_strip_eos(references, eos_token_id)
        diff = utils.list_strip_eos(diff, eos_token_id)
        cfg = utils.list_strip_eos(cfg, eos_token_id)

        for example in range(len(references)):
            hwords = tokenizer.convert_ids_to_tokens(hypotheses[example])
            rwords = tokenizer.convert_ids_to_tokens(references[example])
            dwords = tokenizer.convert_ids_to_tokens(diff[example])
            cwords = tokenizer.convert_ids_to_tokens(cfg[example])

            hwords = tx.utils.str_join(hwords).replace(" ##", "")
            rwords = tx.utils.str_join(rwords).replace(" ##", "")
            dwords = tx.utils.str_join(dwords).replace(" ##", "").replace("[CLS]", "")
            cwords = tx.utils.str_join(cwords).replace(" ##", "").replace("[CLS]", "")

            f1.write(rwords + '\n')
            f2.write(hwords + '\n')
            f3.write(dwords + '\n')
            f4.write(cwords + '\n')


tx.utils.maybe_create_dir(model_dir)
logging_file = os.path.join(model_dir, "logging.txt")
logger = utils.get_logger(logging_file)

# with tf.device('/gpu:0'):
with tf.Session() as sess:  # config=config
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    smry_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    if run_mode == 'train_and_evaluate':
        logger.info('Begin running with train_and_evaluate mode')

        if tf.train.latest_checkpoint(model_dir) is not None:
            logger.info('Restore latest checkpoint in %s' % model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        iterator.initialize_dataset(sess)

        # iterator.restart_dataset(sess, 'test')
        step = _eval_epoch(sess, 0, 'test')

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))
