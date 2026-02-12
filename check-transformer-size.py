# Check Transformer model size
from transformer import Transformer
import tensorflow as tf

transformer = Transformer(
    n_layer=2, d_model=128, d_head=32, n_head=8, d_inner=256,
    d_head_softmax=16, n_head_softmax=8, dropout=0.05,
    n_classes=256, conv_kernel_size=3, n_conv_layer=2, pool_size=20,
    d_kernel_map=512, beta_hat_2=150, model_normalization='preLC',
    head_initialization='forward', softmax_attn=True, output_attn=False
)

dummy = tf.zeros((1, 10000))
_ = transformer(dummy, training=False)
print(f"🔷 Transformer: {transformer.count_params():,} parameters")