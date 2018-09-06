import tensorflow as tf
import tensorflow.contrib as tc

def attend_pooling(pooling_vectors, ref_vector, hidden_size, scope=None):
    """
    Applies attend pooling to a set of vectors according to a reference vector.
    Args:
        pooling_vectors: the vectors to pool
        ref_vector: the reference vector
        hidden_size: the hidden size for attention function
        scope: score name
    Returns:
        the pooled vector
    """
    with tf.variable_scope(scope or 'attend_pooling'):
        U = tf.tanh(tc.layers.fully_connected(pooling_vectors, num_outputs=hidden_size,
                                              activation_fn=None, biases_initializer=None)
                    + tc.layers.fully_connected(tf.expand_dims(ref_vector, 1),
                                                num_outputs=hidden_size,
                                                activation_fn=None))
        logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
        scores = tf.nn.softmax(logits, 1)
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
    return pooled_vector

class GlobalAttentionNetwork(object):
    """
    Implements the Global Attention Network
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def decode(self, passage_vectors, query_vectors):
        """
        Use Global Attention Network to compute the final label
        Args:
            passage_vectors: the encoded passage vectors
            query_vectors: the encoded query vectors
        Returns:
            the label prob
        """
        with tf.variable_scope('ga_decoder'):
            p_random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]),
                                               trainable=True, name="p_random_attn_vector")
            pooled_passage_rep = tc.layers.fully_connected(
                attend_pooling(passage_vectors, p_random_attn_vector, self.hidden_size),
                num_outputs=self.hidden_size, activation_fn=None
            )

            q_random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]),
                                               trainable=True, name="q_random_attn_vector")
            pooled_query_rep = tc.layers.fully_connected(
                attend_pooling(query_vectors, q_random_attn_vector, self.hidden_size),
                num_outputs=self.hidden_size, activation_fn=None
            )

            concat_rep = tf.concat([pooled_passage_rep, pooled_query_rep], 1)

            # binary-classification problem
            logits = tc.layers.fully_connected(concat_rep, num_outputs=2, activation_fn=None)
            return logits
