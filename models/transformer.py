import tensorflow as tf
from .fast_attention import SelfAttention
from .normalization import LayerScaling, LayerCentering
from tensorflow.keras.layers import BatchNormalization as SyncBatchNormalization

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

class PositionalFeature(tf.keras.layers.Layer):
    def __init__(self, d_feature, beta_hat_2, **kwargs):
        super().__init__(**kwargs)
        self.slopes = (tf.range(d_feature, 0, -4.0, dtype=tf.float32) / d_feature) * beta_hat_2

    def call(self, slen, bsz=None):
        pos_seq = tf.range(0, slen, 1.0, dtype=tf.float32)
        n_slopes = (1. / max(float(slen-1), 1.0)) * self.slopes
        fwd = tf.einsum("i,j->ij", pos_seq, n_slopes)
        bwd = tf.reverse(fwd, axis=[0])
        pos_ft = tf.concat([fwd, bwd, -fwd, -bwd], -1)
        s_ft = float(slen-1) * tf.reshape(tf.concat([n_slopes, -n_slopes, -n_slopes, n_slopes], 0), [1, -1])
        if bsz is not None:
            return tf.tile(pos_ft[None, :, :], [bsz, 1, 1]), tf.tile(s_ft[None, :, :], [bsz, 1, 1])
        return pos_ft[None, :, :], s_ft[None, :, :]

class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = tf.keras.layers.Dense(d_inner, activation='relu')
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        self.layer_2 = tf.keras.layers.Dense(d_model)
        self.drop_2 = tf.keras.layers.Dropout(dropout)

    def call(self, inp, training=False):
        x = self.drop_1(self.layer_1(inp), training=training)
        return [self.drop_2(self.layer_2(x), training=training)]

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, n_head, d_head, d_model, d_inner, dropout, feature_map_type, normalize_attn, d_kernel_map, model_normalization, head_init_range, **kwargs):
        super().__init__(**kwargs)
        self.model_normalization = model_normalization
        self.self_attn = SelfAttention(d_model, d_head, n_head, dropout, feature_map_type, normalize_attn, d_kernel_map, head_init_range)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        if model_normalization in ['preLC', 'postLC']:
            self.lc1, self.lc2 = LayerCentering(), LayerCentering()

    def call(self, inputs, training=False):
        inp, p_ft, p_s = inputs
        x = self.lc1(inp) if self.model_normalization == 'preLC' else inp
        attn_out = self.self_attn(x, p_ft, p_s, training=training)
        x = attn_out[0] + inp
        if self.model_normalization == 'postLC': x = self.lc1(x)
        ff_in = self.lc2(x) if self.model_normalization == 'preLC' else x
        ff_out = self.pos_ff(ff_in, training=training)
        res = ff_out[0] + x
        if self.model_normalization == 'postLC': res = self.lc2(res)
        return [res] + attn_out[1:]

class SoftmaxAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_head, **kwargs):
        super().__init__(**kwargs)
        self.d_head, self.n_head = d_head, n_head
        self.q_heads = self.add_weight(shape=(d_head, n_head), name="q_heads")
        self.k_net = tf.keras.layers.Dense(d_head * n_head)
        self.v_net = tf.keras.layers.Dense(d_head * n_head)
        self.scale = 1. / (d_head ** 0.5)

    def build(self, input_shape):
        self.smoothing = self.add_weight(name="smoothing", shape=(), initializer="zeros", trainable=False)

    def call(self, inp, softmax_attn_smoothing=1.0, training=False):
        bsz, slen = shape_list(inp)[:2]
        if training: self.smoothing.assign(softmax_attn_smoothing)
        k = tf.reshape(self.k_net(inp), [-1, slen, self.d_head, self.n_head])
        v = tf.reshape(self.v_net(inp), [-1, slen, self.d_head, self.n_head])
        prob = tf.nn.softmax(tf.einsum("bndh,dh->bnh", k, self.q_heads) * self.scale * self.smoothing, axis=1)
        return tf.reshape(tf.einsum("bndh,bnh->bnhd", v, prob), [bsz, slen, -1]), prob

class Transformer(tf.keras.Model):
    def __init__(self, n_layer, d_model, d_head, n_head, d_inner, d_head_softmax, n_head_softmax, dropout, n_classes, conv_kernel_size, n_conv_layer, pool_size, d_kernel_map, beta_hat_2, model_normalization, head_initialization='forward', softmax_attn=True, output_attn=False):
        super(Transformer, self).__init__()
        self.n_conv_layer, self.pool_size, self.softmax_attn, self.output_attn = n_conv_layer, pool_size, softmax_attn, output_attn
        filters = [min(8*2**i, d_model) for i in range(n_conv_layer-1)] + [d_model]
        self.convs = [tf.keras.layers.Conv1D(filters[l], 11 if l==0 else conv_kernel_size) for l in range(n_conv_layer)]
        self.relus = [tf.keras.layers.ReLU() for _ in range(n_conv_layer)]
        self.pools = [tf.keras.layers.AveragePooling1D(pool_size, pool_size) for _ in range(n_conv_layer)]
        self.pos_feature = PositionalFeature(d_model, beta_hat_2)
        
        ranges = []
        for i in range(n_layer):
            val = 0.5 if i == 0 else 1.0
            if head_initialization == 'forward': ranges.append((0., val))
            elif head_initialization == 'backward': ranges.append((-val, 0.))
            else: ranges.append((-val, val))

        self.trans = [TransformerLayer(n_head, d_head, d_model, d_inner, dropout, 'fourier', False, d_kernel_map, model_normalization, ranges[i]) for i in range(n_layer)]
        self.out_drop = tf.keras.layers.Dropout(dropout)
        if softmax_attn: self.out_attn = SoftmaxAttention(d_model, n_head_softmax, d_head_softmax)
        self.fc = tf.keras.layers.Dense(n_classes)

    def call(self, inputs, softmax_attn_smoothing=1, training=False):
        x = tf.expand_dims(inputs, -1)
        for l in range(self.n_conv_layer):
            x = self.pools[l](self.relus[l](self.convs[l](x)))
        bsz, slen = shape_list(x)[:2]
        p_ft, p_s = self.pos_feature(slen=slen, bsz=bsz)
        
        core = x
        for layer in self.trans:
            core = layer([core, p_ft, p_s], training=training)[0]
        
        core = self.out_drop(core, training=training)
        score = None
        if self.softmax_attn:
            core, score = self.out_attn(core, softmax_attn_smoothing=softmax_attn_smoothing, training=training)
        
        out = self.fc(tf.reduce_mean(core, axis=1))
        return [out, score] if self.output_attn else [out]