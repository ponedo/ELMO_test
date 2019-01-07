import tensorflow as tf
import os
import h5py
from config import *
from bilm import dump_token_embeddings, LanguageModel, \
	load_options_latest_checkpoint


if __name__=="__main__":
    # Dump the token embeddings to a file. Run this once for your dataset.
    # print("Dumping context-independent token embeddings... ")
    # dump_token_embeddings(
    #     whole_vocab_file,
    #     bilm_options_file,
    #     bilm_weight_file,
    #     token_embedding_file)

    options, ckpt_file = load_options_latest_checkpoint(
        os.path.join(rootdir, 'build', 'checkpoints'))

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = 1
            model = LanguageModel(test_options, False)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        print("HELLO!")
        embed = sess.run(model.embedding_weights)
        with h5py.File(token_embedding_file, 'w') as fout:
            fout.create_dataset('embedding', embed.shape, dtype='float32', data=embed)
        print("HELLO AGAIN!")
