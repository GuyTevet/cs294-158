import tensorflow as tf
import ops
import numpy as np

class pixelCNN(object):

    def __init__(self, batch_size=128, image_size=28, num_channels=3, num_colors=4, net_width = 128):

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_colors = num_colors

        self.net_width = net_width

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def add_placeholders(self):
        self.inputs_batch = tf.placeholder(tf.uint8, shape=(None, self.image_size, self.image_size, self.num_channels), name='inputs_batch')
        self.input_float = tf.cast(self.inputs_batch, tf.float32) / tf.constant(self.num_colors - 1, dtype=tf.float32)
        self.input_float = self.input_float - 0.5 #de-bias
        self.input_one_hot = tf.one_hot(self.inputs_batch, depth=self.num_colors)

    def create_feed_dict(self, inputs_batch):
        return {self.inputs_batch: inputs_batch}

    def add_net(self):

        net = ops.conv_masked(self.input_float, 'conv7x7', 7, in_channels=self.num_channels, type_A=True)

        for i in range(12):
            net = ops.res_block(net, scope='res_block_{}'.format(i+1),channels=self.net_width)

        net = tf.nn.relu(net)
        net = ops.conv_1x1(net, 'final_conv1x1_1',
                       in_channels=self.net_width,
                       out_channels=self.net_width)

        net = tf.nn.relu(net)
        net = ops.conv_1x1(net, 'final_conv1x1_2',
                       in_channels=self.net_width,
                       out_channels=self.num_channels * self.num_colors)

        net = tf.reshape(net, (-1, self.image_size, self.image_size, self.num_channels, self.num_colors))

        return net

    def add_loss_op(self, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_one_hot, logits=logits))

    def add_inference_op(self, logits):
        return tf.nn.softmax(logits, axis=4, name='inference')

    def add_training_op(self, loss):
        return tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    def add_summary_op(self, pred, loss, name):
        """Sets up the summary Op.

        Generates summaries about the model to be displayed by TensorBoard.
        https://www.tensorflow.org/api_docs/python/tf/summary
        for more information.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
            loss: Loss tensor (a scalar).
        Returns:
            summary: training records summary.
        """
        tf.summary.scalar('loss', loss)
        # tf.summary.scalar('accuracy', accuracy)

        return tf.summary.merge_all()


    def train_on_batch(self, sess, inputs_batch, summarize):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch)
        if summarize:
            _, loss, summary, pred = sess.run([self.train_op, self.loss, self.summary, self.pred], feed_dict=feed)
            return loss, summary, pred
        else:
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            return loss

    def add_grads(self):
        # receptive field experiment
        dy = self.logits[0,14,14,0] # mid output
        dx = self.input_float # with respect to input
        grad = tf.gradients(dy,dx) # calc grad
        return grad

    def calc_grad(self, sess):
        feed = self.create_feed_dict(np.random.choice(4, size=(1, self.image_size, self.image_size, self.num_channels)).astype(np.uint8))
        grad = sess.run(self.grad, feed_dict=feed)
        return grad[0]

    def sample(self, sess, num_images=1):
        images = np.random.choice(4, size=(num_images, self.image_size, self.image_size, self.num_channels)).astype(np.uint8)

        for i in range(28):
            for j in range(28):
                for k in range(3):
                    pred = sess.run(self.pred,{self.inputs_batch: images})
                    for b in range(num_images):
                        images[b, i, j, k] = np.random.choice(4, p=pred[b, i, j, k])

        return images

    def predict_on_batch(self, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions

    def add_measurements(self, pred):

        pass

        # pred_arg = tf.argmax(pred,axis=1,output_type=tf.int32)
        # accuracy = tf.reduce_mean(tf.cast(tf.math.equal(pred_arg, self.labels),tf.float32))
        # return accuracy


    def build(self, name='', summarize=False):

        with self.graph.as_default():
            with tf.variable_scope(name):
                self.add_placeholders()
                self.logits = self.add_net()
                self.loss = self.add_loss_op(self.logits)
                self.pred = self.add_inference_op(self.logits)
                self.grad = self.add_grads()
                # self.accuracy = self.add_measurements(self.pred)
                self.train_op = self.add_training_op(self.loss)
                if summarize:
                    self.summary = self.add_summary_op(self.pred, self.loss, name)