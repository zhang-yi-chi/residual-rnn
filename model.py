from __future__ import print_function
import tensorflow as tf


ENCODER_INPUT_DIM = 32
ENCODER1_DIM = 64
ENCODER2_DIM = 128
ENCODER3_DIM = 128
DECODER_INPUT_DIM = 128
DECODER1_DIM = 128
DECODER2_DIM = 128
DECODER3_DIM = 64
DECODER4_DIM = 32


def rnn_conv(name, inputs, hiddens, filters, kernel_size, strides):
    '''Convolution RNN cell

    See detail in formula (4) in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf

    Args:
        name: name of current Conv RNN layer
        inputs: inputs tensor with shape (batch_size, height, width, channel)
        hiddens: hidden states from the previous iteration
        kernel_size: tuple of kernel size
        strides: strides size

    Output:
        hidden state and cell state of this layer
    '''
    gates_filters = 4 * filters
    hidden, cell = hiddens
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_inputs = tf.layers.conv2d(inputs=inputs, filters=gates_filters,
                                       kernel_size=kernel_size, strides=strides, padding='same', name='conv_inputs')
        conv_hidden = tf.layers.conv2d(inputs=hidden, filters=gates_filters,
                                       kernel_size=kernel_size, padding='same', name='conv_hidden')
    in_gate, f_gate, out_gate, c_gate = tf.split(
        conv_inputs + conv_hidden, 4, axis=-1)
    in_gate = tf.nn.sigmoid(in_gate)
    f_gate = tf.nn.sigmoid(f_gate)
    out_gate = tf.nn.sigmoid(out_gate)
    c_gate = tf.nn.tanh(c_gate)
    new_cell = tf.multiply(f_gate, cell) + tf.multiply(in_gate, c_gate)
    new_hidden = tf.multiply(out_gate, tf.nn.tanh(new_cell))
    return new_hidden, new_cell


def initial_hidden(input_size, filters, kernel_size, name):
    """Initialize hidden and cell states, all zeros"""
    h_name = name + '_h'
    c_name = name + '_c'
    shape = [input_size] + kernel_size + [filters]
    hidden = tf.zeros(shape)
    cell = tf.zeros(shape)
    return hidden, cell


def padding(x, stride):
    if x % stride == 0:
        return x // stride
    else:
        return x // stride + 1


class encoder(object):
    """Encoder

    See detail in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf

    Args:
        batch_size: mini-batch size
        is_training: boolean variable controls binarizer behaviour
        height: height of input image data
        width: width of input image data
    """

    def __init__(self, batch_size, is_training=False, height=32, width=32):
        self.is_training = is_training
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        """Initialize hidden and cell states"""
        height = padding(padding(self.height, 2), 2)
        width = padding(padding(self.width, 2), 2)
        self.hiddens1 = initial_hidden(
            self.batch_size, ENCODER1_DIM, [height, width], 'encoder1')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens2 = initial_hidden(
            self.batch_size, ENCODER2_DIM, [height, width], 'encoder2')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens3 = initial_hidden(
            self.batch_size, ENCODER3_DIM, [height, width], 'encoder3')

    def encode(self, inputs):
        """Compress inputs into a vector of 128 lengths with value {-1, 1}"""
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_rnn_input = tf.layers.conv2d(
                inputs=inputs, filters=ENCODER_INPUT_DIM, kernel_size=[3, 3], strides=(2, 2), padding='same', name='encoder_rnn_input')
            self.hiddens1 = rnn_conv('encoder_rnn_conv_1',
                                     encoder_rnn_input, self.hiddens1, ENCODER1_DIM, [3, 3], (2, 2))
            self.hiddens2 = rnn_conv('encoder_rnn_conv_2',
                                     self.hiddens1[0], self.hiddens2, ENCODER2_DIM, [3, 3], (2, 2))
            self.hiddens3 = rnn_conv('encoder_rnn_conv_3',
                                     self.hiddens2[0], self.hiddens3, ENCODER3_DIM, [3, 3], (2, 2))
        code = self.binarizer(self.hiddens3[0])
        return code

    def binarizer(self, inputs, filters=32, kernel_size=(1, 1)):
        with tf.variable_scope('binarizer', reuse=tf.AUTO_REUSE):
            binarizer_input = tf.layers.conv2d(inputs=inputs, filters=filters,
                                               kernel_size=kernel_size, padding='same', name='binarizer_inputs', activation=tf.nn.tanh)
        if self.is_training:
            probs = (1 + binarizer_input) / 2
            dist = tf.distributions.Bernoulli(probs=probs, dtype=tf.float32)
            noise = 2 * dist.sample(name='noise') - 1 - binarizer_input
            output = binarizer_input + tf.stop_gradient(noise)
            # output = binarizer_input + noise
        else:
            output = tf.sign(binarizer_input)

        return output


class decoder(object):
    """Decoder

    See detail in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf

    Args:
        batch_size: mini-batch size
        height: height of input image data
        width: width of input image data
    """

    def __init__(self, batch_size, height=32, width=32):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        height = padding(self.height, 2)
        width = padding(self.width, 2)
        self.hiddens4 = initial_hidden(
            self.batch_size, DECODER4_DIM, [height, width], 'decoder4')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens3 = initial_hidden(
            self.batch_size, DECODER3_DIM, [height, width], 'decoder3')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens2 = initial_hidden(
            self.batch_size, DECODER2_DIM, [height, width], 'decoder2')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens1 = initial_hidden(
            self.batch_size, DECODER1_DIM, [height, width], 'decoder1')

    def decode(self, inputs):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_rnn_input = tf.layers.conv2d(inputs=inputs, filters=DECODER_INPUT_DIM, kernel_size=[
                                                 3, 3], strides=(1, 1), padding='same', name='decoder_rnn_input')
            self.hiddens1 = rnn_conv('decoder_rnn_conv_1',
                                     decoder_rnn_input, self.hiddens1, DECODER1_DIM, [2, 2], (1, 1))
            d_rnn_h1 = tf.depth_to_space(self.hiddens1[0], 2)
            self.hiddens2 = rnn_conv('decoder_rnn_conv_2',
                                     d_rnn_h1, self.hiddens2, DECODER2_DIM, [3, 3], (1, 1))
            d_rnn_h2 = tf.depth_to_space(self.hiddens2[0], 2)
            self.hiddens3 = rnn_conv('decoder_rnn_conv_3',
                                     d_rnn_h2, self.hiddens3, DECODER3_DIM, [3, 3], (1, 1))
            d_rnn_h3 = tf.depth_to_space(self.hiddens3[0], 2)
            self.hiddens4 = rnn_conv('decoder_rnn_conv_4',
                                     d_rnn_h3, self.hiddens4, DECODER4_DIM, [3, 3], (1, 1))
            d_rnn_h4 = tf.depth_to_space(self.hiddens4[0], 2)

            output = tf.layers.conv2d(inputs=d_rnn_h4, filters=3, kernel_size=[
                                      3, 3], strides=(1, 1), padding='same', name='output', activation=tf.nn.tanh)
        return output / 2


class ResidualRNN(object):
    """Encoder

    See detail in paper
    "Full Resolution Image Compression with Recurrent Neural Networks"
    https://arxiv.org/pdf/1608.05148.pdf

    Args:
        batch_size: mini-batch size
        num_iters: number of iterations
    """

    def __init__(self, batch_size, num_iters):
        self.batch_size = batch_size
        self.num_iters = num_iters

        self.encoder = encoder(batch_size, is_training=True)
        self.decoder = decoder(batch_size)
        self.inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.build_graph()

    def build_graph(self):
        inputs = self.inputs / 255.0 - 0.5
        self.loss = 0
        self.compress = tf.zeros_like(inputs) + 0.5
        self.encoder.init_hidden()
        self.decoder.init_hidden()
        for i in range(self.num_iters):
            code = self.encoder.encode(inputs)
            outputs = self.decoder.decode(code)
            self.compress += outputs
            self.loss += tf.losses.absolute_difference(inputs, outputs)
            inputs = inputs - outputs
        self.loss /= self.batch_size  # * 32 * 32 * 3 * self.num_iters
        self.compress *= 255

    def get_loss(self):
        return self.loss

    def get_compress(self):
        return self.compress

    def debug(self):
        code = self.encoder.encode(self.inputs)
        output = self.decoder.decode(code)
        print(output.get_shape())


if __name__ == '__main__':
    model = ResidualRNN(128, 10)
    model.debug()
