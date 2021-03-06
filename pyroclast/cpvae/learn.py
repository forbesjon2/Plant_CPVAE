import os

import numpy as np
import tensorflow as tf

from pyroclast.common.util import img_postprocess
from pyroclast.common.tf_util import calculate_accuracy, run_epoch_ops
from pyroclast.cpvae.cpvae import CpVAE
from pyroclast.cpvae.ddt import transductive_box_inference, get_decision_tree_boundaries
from pyroclast.cpvae.tf_models import Encoder, Decoder
from pyroclast.common.util import dummy_context_mgr
from pyroclast.cpvae.util import update_model_tree, build_model
from gen_tf_dataset import gen_data_dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_data = gen_data_dict()

def learn(
        data_dict,
        seed=None,
        latent_dim=128,
        epochs=500,
        image_size=128,
        max_tree_depth=5,
        max_tree_leaf_nodes=16,
        tree_update_period=10, #label_attr='No_Beard',
        optimizer='adam',  # adam or rmsprop
        learning_rate=1e-3,
        classification_coeff=1.,
        distortion_fn='disc_logistic',  # disc_logistic or l2
        output_dir='./',
        load_dir=None,
        num_samples=5):
    del seed  # currently unused
    #num_classes = data_dict['num_classes']
    num_classes = 2

    print('setup model')
    model, optimizer, global_step = build_model(
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        num_classes=num_classes,
        latent_dim=latent_dim,
        image_size=image_size,
        max_tree_depth=max_tree_depth,
        max_tree_leaf_nodes=max_tree_leaf_nodes)

    print('checkpointing and tensorboard')
    writer = tf.summary.create_file_writer(output_dir)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(checkpoint,
                                              directory=os.path.join(
                                                  output_dir, 'model'),
                                              max_to_keep=3,
                                              keep_checkpoint_every_n_hours=2)
    # reload if data exists
    if load_dir:
        status = checkpoint.restore(tf.train.latest_checkpoint(str(load_dir)))
        print("load: ", status.assert_existing_objects_matched())

    # define minibatch fn
    def run_minibatch(epoch, batch, is_train=True):
        x = tf.cast(batch['image'], tf.float32)
        #print('x:', x)
        #labels = tf.cast(batch['attributes'][label_attr], tf.int32)
        labels = tf.cast(batch['label'], tf.int32)
        #print('labels:', labels)

        with tf.GradientTape() if is_train else dummy_context_mgr() as tape:
            x_hat, y_hat, z_posterior = model(x)
            y_hat = tf.cast(y_hat, tf.float32)  # from double to single fp
            distortion, rate = model.vae_loss(x,
                                              x_hat,
                                              z_posterior,
                                              distortion_fn=distortion_fn,
                                              y=labels)
            classification_loss = classification_coeff * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=y_hat)
            loss = tf.reduce_mean(distortion + rate + classification_loss)

        # calculate gradients for current loss
        if is_train:
            global_step.assign_add(1)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        with writer.as_default():
            prediction = tf.math.argmax(y_hat, axis=1, output_type=tf.int32)
            classification_rate = tf.reduce_mean(
                tf.cast(tf.equal(prediction, labels), tf.float32))
            tf.summary.scalar("distortion",
                              tf.reduce_mean(distortion),
                              step=global_step)
            tf.summary.scalar("rate", tf.reduce_mean(rate), step=global_step)
            tf.summary.scalar("classification_loss",
                              tf.reduce_mean(classification_loss),
                              step=global_step)
            tf.summary.scalar("classification_rate",
                              classification_rate,
                              step=global_step)
            tf.summary.scalar("sum_loss", loss, step=global_step)

    print('run training loop')
    #update_model_tree(data_dict['train'],model,epoch='init',label_attr=label_attr,output_dir=output_dir,limit=10)
    update_model_tree(data_dict['train'],model,epoch='init',output_dir=output_dir,limit=10)
    for epoch in range(epochs):
        print('epoch: %s'%epoch)
        print('train')
        for batch in data_dict['train']:
            run_minibatch(epoch, batch, is_train=True)

        print('test')
        for batch in data_dict['test']:
            run_minibatch(epoch, batch, is_train=False)

        print('save and update')
        ckpt_manager.save(checkpoint_number=epoch)
        if epoch % tree_update_period == 0:
            #update_model_tree(data_dict['train'], model, epoch, label_attr, output_dir)
            update_model_tree(data_dict['train'], model, epoch, output_dir)

        print('generate sample images')
        for i in range(num_samples):
            im = img_postprocess(np.squeeze(model.sample()))
            im.save(
                os.path.join(output_dir,
                             "epoch_{}_sample_{}.png".format(epoch, i)))

if __name__ == "__main__":
    import sys
    if len(sys.argv)==1:
        print('disc_logistic or l2')
    else:
        learn(test_data, distortion_fn=sys.argv[1])

