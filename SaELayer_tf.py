import ipdb
import tensorflow as tf
import numpy as np
#tf 1.*
def SE_layer(inputs, reduction=16,scope='se'):
    with tf.variable_scope(scope):
        _,H,W,C = inputs.get_shape().as_list()
        pooled = tf.layers.average_pooling2d(inputs, pool_size=(inputs.shape[1],inputs.shape[2]), strides=1)
        se = tf.reshape(pooled, [-1, C])
        se = tf.layers.dense(se, units=C//reduction, activation=tf.nn.relu)
        se = tf.layers.dense(se, units=C, activation=tf.nn.sigmoid)
        se_reshape = tf.reshape(se, [-1, 1, 1, C])
        outputs = tf.multiply(inputs, se_reshape)
    return outputs


def SaElayer(inputs, reduction=16, scope='sae'):
    with tf.variable_scope(scope):
        _,H,W,C = inputs.get_shape().as_list()
        pooled = tf.layers.average_pooling2d(inputs, pool_size=(H,W), strides=1)#（B，1，1，C)
        se = tf.reshape(pooled, [-1, C])
        sae_fc1 = tf.layers.dense(se, units=C // reduction, activation=tf.nn.relu)
        sae_fc2 = tf.layers.dense(se, units=C // reduction, activation=tf.nn.relu)
        sae_fc3 = tf.layers.dense(se, units=C // reduction, activation=tf.nn.relu)
        sae_fc4 = tf.layers.dense(se, units=C // reduction, activation=tf.nn.relu)
        sae_concate = tf.concat([sae_fc1, sae_fc2, sae_fc3, sae_fc4], axis=1)

        sae = tf.layers.dense(sae_concate, units=C, activation=tf.nn.sigmoid)
        sae_reshape = tf.reshape(sae, [-1, 1, 1, C])
        outputs = tf.multiply(inputs, sae_reshape)

    return outputs

if __name__ == '__main__':
    inputs = tf.placeholder(dtype=tf.float32,shape=(4,32,32,16),name='inputs')
    data = np.random.random((4,32,32,16))
    # mode = squeeze_excitement_layer(inputs)
    mode = SaElayer(inputs)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        mdoel = sess.run(mode,feed_dict={inputs:data})
        ipdb.set_trace()
        print(mdoel.shape)

