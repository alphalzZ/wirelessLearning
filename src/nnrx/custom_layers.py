import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Dropout, Dense, Layer, Conv2D, Conv2DTranspose, LayerNormalization, SeparableConv2D,GroupNormalization,BatchNormalization

from tensorflow.keras import Model
from tensorflow.nn import relu

class SeparableConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 depth_multiplier=1,
                 activation=None,
                 use_bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        if padding.lower() not in {"valid", "same"}:
            raise ValueError("padding must be 'valid' or 'same'")
        self.padding = padding.lower()
        self.depth_multiplier = int(depth_multiplier)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    # ---------- build ----------
    def build(self, input_shape):
        if input_shape.ndims != 4:
            raise ValueError("Input must be NHWC rank‑4")
        self.in_channels = int(input_shape[-1])
        k_h, k_w = self.kernel_size

        # Depth‑wise kernel   (k,k,Cin,depth_multiplier)
        self.depthwise_kernel = self.add_weight(
            "depthwise_kernel",
            shape=(k_h, k_w, self.in_channels, self.depth_multiplier),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Point‑wise kernel   (1,1,Cin*depth_multiplier,Cout)
        self.pointwise_kernel = self.add_weight(
            "pointwise_kernel",
            shape=(1, 1, self.in_channels * self.depth_multiplier, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    # ---------- output‑shape helper ----------
    def _deconv_dim(self, dim_in, k, s):
        if self.padding == "valid":
            return (dim_in - 1) * s + k
        return dim_in * s            # 'same'

    # ---------- call ----------
    def call(self, inputs):
        bs, h_in, w_in = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        h_out = self._deconv_dim(h_in, self.kernel_size[0], self.strides[0])
        w_out = self._deconv_dim(w_in, self.kernel_size[1], self.strides[1])

        # 1) true depth‑wise de‑conv
        out_shape = tf.stack([bs, h_out, w_out,
                              self.in_channels * self.depth_multiplier])
        depthwise_out = tf.nn.depthwise_conv2d_backprop_input(
            input_sizes=out_shape,
            filter=self.depthwise_kernel,
            out_backprop=inputs,                     # “gradients” == activations
            strides=[1, *self.strides, 1],
            padding=self.padding.upper(),            # 'VALID' / 'SAME'
            data_format="NHWC",
        )                                            # --> (N,Hout,Wout,Cin*D)

        # 2) point‑wise projection
        outputs = tf.nn.conv2d(
            depthwise_out,
            filters=self.pointwise_kernel,
            strides=1,
            padding="SAME",
            data_format="NHWC",
        )
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    # ---------- shape inference ----------
    def compute_output_shape(self, input_shape):
        h_out = self._deconv_dim(input_shape[1], self.kernel_size[0], self.strides[0])
        w_out = self._deconv_dim(input_shape[2], self.kernel_size[1], self.strides[1])
        return (input_shape[0], h_out, w_out, self.filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        depth_multiplier=self.depth_multiplier,
                        activation=tf.keras.activations.serialize(self.activation),
                        use_bias=self.use_bias))
        return cfg
    

class SeparableConv2DTransposeONNX(tf.keras.layers.Layer):
    """Depth‑wise‑Separable Conv2DTranspose, ONNX‑friendly."""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="valid",
                 depth_multiplier=1,
                 activation=None,
                 use_bias=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")

        if padding.lower() not in {"valid", "same"}:
            raise ValueError("padding must be 'valid' or 'same'")
        self.padding = padding.lower()
        self.depth_multiplier = int(depth_multiplier)

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    # ------------------------------------------------------------ #
    def build(self, input_shape):
        if input_shape.ndims != 4:
            raise ValueError("input must be rank‑4 NHWC")

        self.c_in = int(input_shape[-1])
        k_h, k_w = self.kernel_size

        # depth‑wise kernel  (k,k,Cin,D)
        self.dw_kernel = self.add_weight(
            "depthwise_kernel",
            shape=(k_h, k_w, self.c_in, self.depth_multiplier),
            initializer="glorot_uniform",
            trainable=True,
        )
        # point‑wise kernel  (1,1,Cin*D,Cout)
        self.pw_kernel = self.add_weight(
            "pointwise_kernel",
            shape=(1, 1, self.c_in * self.depth_multiplier, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias", shape=(self.filters,), initializer="zeros", trainable=True
            )
        else:
            self.bias = None
        super().build(input_shape)

    # ------------------------------------------------------------ #
    @staticmethod
    def _out_dim(length, k, s, padding):
        return (length - 1) * s + k if padding == "valid" else length * s

    # ------------------------------------------------------------ #
    def _make_group_kernel(self):
        """
        将 (k,k,Cin,D) → (k,k,Cin*D,Cin) 并置零跨通道权重，
        使 conv2d_transpose 等价于 depth‑wise groups＝Cin。
        """
        k_h, k_w = self.kernel_size

        # (k,k,Cin,D)  ->  (k,k,D,Cin)
        ker = tf.transpose(self.dw_kernel, [0, 1, 3, 2])
        # (k,k,D,Cin)  ->  (k,k,D,Cin,1)
        ker = tf.expand_dims(ker, -1)                     # for broadcasting
        # diag mask  (1,1,1,Cin,Cin)
        diag = tf.reshape(tf.eye(self.c_in, dtype=ker.dtype),
                          [1, 1, 1, self.c_in, self.c_in])
        ker = ker * diag                                  # zero cross‑channel
        ker = tf.reshape(ker,
                         [k_h, k_w,
                          self.c_in * self.depth_multiplier,
                          self.c_in])                     # (k,k,Cout,Cin)
        return ker

    # ------------------------------------------------------------ #
    def call(self, inputs, training=False):
        bs, h_in, w_in = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        h_out = self._out_dim(h_in, self.kernel_size[0], self.strides[0], self.padding)
        w_out = self._out_dim(w_in, self.kernel_size[1], self.strides[1], self.padding)
        out_shape = tf.stack([bs, h_out, w_out,
                              self.c_in * self.depth_multiplier])

        # 1) depth‑wise (grouped) transpose‑conv via masked full kernel
        dw_full_kernel = self._make_group_kernel()
        x = tf.nn.conv2d_transpose(
            inputs,
            filters=dw_full_kernel,
            output_shape=out_shape,
            strides=[1, *self.strides, 1],
            padding=self.padding.upper(),        # 'VALID' / 'SAME'
            data_format="NHWC",
        )                                        #  (N,Hout,Wout,Cin*D)

        # 2) point‑wise projection
        x = tf.nn.conv2d(
            x, self.pw_kernel,
            strides=1, padding="SAME", data_format="NHWC"
        )
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias, data_format="NHWC")
        if self.activation is not None:
            x = self.activation(x)
        return x

    # ------------------------------------------------------------ #
    def compute_output_shape(self, input_shape):
        h_out = self._out_dim(input_shape[1], self.kernel_size[0],
                              self.strides[0], self.padding)
        w_out = self._out_dim(input_shape[2], self.kernel_size[1],
                              self.strides[1], self.padding)
        return (input_shape[0], h_out, w_out, self.filters)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(filters=self.filters,
                        kernel_size=self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        depth_multiplier=self.depth_multiplier,
                        activation=tf.keras.activations.serialize(self.activation),
                        use_bias=self.use_bias))
        return cfg


class residualBlock(Layer):
    def __init__(self, channels=64, dilation=(1,1), groupNorm=True, sepConv=True):
        super().__init__()
        self.groupNorm = groupNorm
        self.sepConv = sepConv
        self.dilation = dilation
        self.out_channels = channels
        self.channels_per_group = 16
    def build(self, input_shape):
        in_channels = input_shape[-1]
        if self.groupNorm:
            self.normal1 = GroupNormalization(groups=self.out_channels//self.channels_per_group,axis=-1)
            self.normal2 = GroupNormalization(groups=self.out_channels//self.channels_per_group,axis=-1)
        else:
            self.normal1 = LayerNormalization(axis=(-1))
            self.normal2 = LayerNormalization(axis=(-1))
        if self.sepConv:
            self.conv1 = SeparableConv2D(self.out_channels, kernel_size=[3, 3], dilation_rate=self.dilation, padding='same', activation=None)
            self.conv2 = SeparableConv2D(self.out_channels, kernel_size=[3, 3], dilation_rate=self.dilation, padding='same', activation=None)
        else:
            self.conv1 = Conv2D(self.out_channels, kernel_size=[3, 3], dilation_rate=self.dilation, padding='same', activation=None)
            self.conv2 = Conv2D(self.out_channels, kernel_size=[3, 3], dilation_rate=self.dilation, padding='same', activation=None)
        if in_channels != self.out_channels:
            # 用 1×1 卷积把通道数对齐
            # self.proj = SeparableConv2D(self.out_channels, kernel_size=[1, 1], use_bias=False)
            self.proj = Conv2D(self.out_channels, kernel_size=[1, 1], use_bias=False)
        else:
            self.proj = tf.identity   # 直接返回输入
        super().build(input_shape)
    def call(self, inputs):
        z = self.normal1(inputs)
        z = relu(z)
        z = self.conv1(z)
        z = self.normal2(z)
        z = relu(z)
        z = self.conv2(z)
        z = z + self.proj(inputs)
        return z
    
class NNrecevier(Model):
    def __init__(self,num_bits_per_symbol, groupNorm=True, sepConv=True,transpose=True):
        super().__init__()
        self.num_bits_per_symbol = num_bits_per_symbol
        self.groupNorm = groupNorm
        self.sepConv = sepConv
        self.num_res_blok = 11
        self.dilation = [(1,1),(1,1),(2,3),(2,3),(2,3),(3,6),(2,3),(2,3),(2,3),(1,1),(1,1)]
        self.out_channels = [64,64,128,128,256,256,256,128,128,64,64]
        self.transpose = transpose
    def build(self, input_shape):
        self.init_norm = GroupNormalization(groups=2,axis=-1)
        if self.transpose:
            self.conv1 = Conv2DTranspose(64, kernel_size=[3,3], strides = [1,1], padding='valid')
        else:
            self.conv1 = Conv2D(64, kernel_size=[3, 3], dilation_rate=(1,1), padding='same', activation=None)
        self.blocks = []
        for i in range(self.num_res_blok):
            res = residualBlock(self.out_channels[i], self.dilation[i], self.groupNorm, self.sepConv)
            self.blocks.append(res)
        if self.transpose:
            self.outconv = Conv2D(self.num_bits_per_symbol, kernel_size=[3, 3], dilation_rate=(1,1), activation=None)  # max 8bit per symbol(256QAM)
        else:
            self.outconv = Conv2D(self.num_bits_per_symbol, kernel_size=[1, 1], dilation_rate=(1,1), activation=None)
    def call(self, inputs):
        z = self.init_norm(inputs)
        z = self.conv1(z)
        for i in range(self.num_res_blok):
            z = self.blocks[i](z)
        z = self.outconv(z)
        return z
## CGNN

# Define the CGNN class here

#1. CNNinit
class CNNinit(Layer):
    def __init__(self, out_channels=54, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, **kwargs):
        super(CNNinit, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
    def build(self, input_shape):
        self.conv1 = SeparableConv2D(128, self.kernel_size, strides=self.strides , dilation_rate=(1,1), padding=self.padding, activation=self.activation)
        self.norm1 = GroupNormalization(groups=4, axis=-1)
        self.conv2 = SeparableConv2D(128, self.kernel_size, strides=self.strides , dilation_rate=(2,3), padding=self.padding, activation=self.activation)
        self.norm2 = GroupNormalization(groups=4, axis=-1)
        self.conv3 = SeparableConv2D(self.out_channels, self.kernel_size, strides=self.strides, dilation_rate=(2,3), padding=self.padding, activation=self.activation)
        self.norm3 = GroupNormalization(groups=2, axis=-1)
        in_channels = input_shape[-1]
        if in_channels != self.out_channels:
            self.proj = SeparableConv2D(self.out_channels, (1, 1), strides=self.strides , padding='same')
        else:
            self.proj = tf.identity
        super(CNNinit, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = relu(x) 
        x = self.conv2(x)
        x = self.norm2(x)
        x = relu(x) 
        x = self.conv3(x)
        x = self.norm3(x)
        return x + self.proj(inputs)

#2.MLPMessgePassing
class MLPMessgePassing(Layer):
    def __init__(self, hidden_dim=128):
        super(MLPMessgePassing, self).__init__()
        self.hidden_dim = hidden_dim
    def build(self,input_shape):
        self.dense1 = Dense(self.hidden_dim)
        self.norm = BatchNormalization()
        self.dense2 = Dense(input_shape[-1])
        super(MLPMessgePassing, self).build(input_shape)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = relu(x)
        x = self.norm(x)
        x = self.dense2(x)
        return x

#3.CNNupdatate
class CNNupdate(Layer):
    def __init__(self, out_channels=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, **kwargs):
        super(CNNupdate, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        
    def build(self, input_shape):
        self.conv1 = SeparableConv2D(128, self.kernel_size, strides=self.strides, dilation_rate=(1,1), padding=self.padding, activation=self.activation)
        self.norm1 = GroupNormalization(groups=8, axis=-1)
        self.conv2 = SeparableConv2D(128, self.kernel_size, strides=self.strides, dilation_rate=(2,3), padding=self.padding, activation=self.activation)
        self.norm2 = GroupNormalization(groups=8, axis=-1)
        self.conv3 = SeparableConv2D(self.out_channels, self.kernel_size, strides=self.strides, dilation_rate=(2,3), padding=self.padding, activation=self.activation)
        self.norm3 = GroupNormalization(groups=2, axis=-1)
        in_channels = input_shape[-1]
        if in_channels != self.out_channels:
            self.proj = SeparableConv2D(self.out_channels, (1, 1), strides=self.strides , padding='same')
        else:
            self.proj = tf.identity
        super(CNNupdate, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = relu(x) 
        x = self.conv2(x)
        x = self.norm2(x)
        x = relu(x) 
        x = self.conv3(x)
        x = x + self.proj(inputs)
        x = self.norm3(x)
        return x


#4. CGNN
class CGNN(Layer):
    def __init__(self, num_layers=8, out_channels=64, multi_head=False, training = True, **kwargs):
        super(CGNN, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.multi_head = multi_head
        self.training = training
    def build(self, input_shape):
        self.layers_list = [(MLPMessgePassing(), CNNupdate(self.out_channels)) for _ in range(self.num_layers)]
        super(CGNN, self).build(input_shape)
    
    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self.num_layers

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= self.num_layers),\
            "Invalid number of iterations"
        self.num_layers = val

    def call(self, inputs):
        outputs = []
        x = inputs
        for i in range(self.num_layers): #TODO 每次循环都需要将x送入llredout，计算损失函数，最后计算总的损失
            msg_pass, cnn_update = self.layers_list[i]
            a = msg_pass(x)
            b = tf.concat([x,a],axis=-1)
            x = cnn_update(b)
            if self.multi_head:
                outputs.append(x) # Store the output of each layer for loss calculation
        if self.multi_head and self.training:
            x = tf.stack(outputs, axis=0)
        return x
    
#5. LLRredout
class LLRredout(Layer):
    def __init__(self, num_bits_per_symbol, **kwargs):
        super(LLRredout, self).__init__(**kwargs)
        self.dense = Dense(num_bits_per_symbol)
    def call(self, inputs):
        x = self.dense(inputs)
        return x

#6. CGNNRecevier
class CGNNRecevier(Model):
    def __init__(self, num_layers=8, num_bits_per_symbol=6, out_channels=128, multi_head=False, training=True, **kwargs):
        super(CGNNRecevier, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.multi_head = multi_head
        self.cnn_init = CNNinit(out_channels)
        self.cggn = CGNN(num_layers, out_channels, multi_head, training)
        self.llr_redout = LLRredout(num_bits_per_symbol)
    
    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self.num_layers

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= self.num_layers),\
            "Invalid number of iterations"
        self.cggn.num_it = val

    def call(self, inputs):
        x = self.cnn_init(inputs)
        x = self.cggn(x)
        x = self.llr_redout(x)
        return x



if __name__ == '__main__':
    import timeit
    x = tf.random.normal((1, 3276, 14, 10))
    cgnnrecevier = CGNNRecevier(num_layers=8, num_bits_per_symbol=8, multi_head=True, training=True)
    start_time = timeit.default_timer()
    output = cgnnrecevier(x)
    print(output.shape)  # Expected shape: (1, num_bits_per_symbol)
    end_time = timeit.default_timer()
    print(f"Running time with default num_it: {end_time - start_time} seconds")
    cgnnrecevier.summary()
    #计算cgnnrecevier.num_it = 1的运行时间
    start_time = timeit.default_timer()
    cgnnrecevier.num_it = 1
    output = cgnnrecevier(x)
    print(output.shape)  # Expected shape: (1, num_bits_per_symbol)
    end_time = timeit.default_timer()
    print(f"Running time with default num_it: {end_time - start_time} seconds")



