from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, LayerNormalization, SeparableConv2D,GroupNormalization
from tensorflow.keras import Model
from tensorflow.nn import relu
from tensorflow.keras import activations, initializers, regularizers, constraints
import tensorflow as tf
import os
import pickle
import numpy as np
from scipy.io import loadmat
import re
import datetime
import tf2onnx
import onnx
print(tf.__version__)
from tensorflow.python.keras.utils import conv_utils
import sys
from pathlib import Path
# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from src.nnrx.custom_layers import SeparableConv2DTransposeONNX, residualBlock, NNrecevier,CGNNRecevier
from src.config import OFDMConfig ,load_config# Import OFDMConfig
from src.nnrx.data_generator import create_tf_dataset # Import create_tf_dataset

class ThreePhaseLR(LearningRateSchedule):
    def __init__(self, target_lr=0.001, total_steps=30000, warmup_steps=800, decay_start=9000):
        super().__init__()
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_start = tf.cast(decay_start, tf.float32)
        self.max_lr = target_lr  # 基础学习率

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # 阶段1：线性预热 (0 → max_lr)
        warmup_lr = self.max_lr * (step / self.warmup_steps)

        # 阶段2：稳定阶段 (max_lr)
        stable_lr = self.max_lr

        # 阶段3：线性衰减 (max_lr → 0)
        decay_steps = self.total_steps - self.decay_start
        decay_lr = self.max_lr * (1 - (step - self.decay_start) / decay_steps)

        # 条件选择
        return tf.case(
            [
                # 预热阶段
                (step < self.warmup_steps, lambda: warmup_lr),
                # 稳定阶段
                (step < self.decay_start, lambda: stable_lr),
                # 衰减阶段（需限制step不超过total_steps）
                (step <= self.total_steps, lambda: decay_lr)
            ],
            default=lambda: tf.constant(0.0, dtype=tf.float32)
        )

class NNrecevierModel(Model):
    def __init__(self, num_bits_per_symbol=6, cgnn_flag=False, multi_head=False, training=True):
        super().__init__()
        if cgnn_flag:
            print('cgnn model!')
            self.nnrecevier = CGNNRecevier(8, num_bits_per_symbol, multi_head = multi_head, training = training)
        else:
            self.nnrecevier = NNrecevier(num_bits_per_symbol)
    # @tf.function()
    def call(self, inputs):
        return self.nnrecevier(inputs)


def loss_function(logits, labels, function,multi_head=False):
    #logits shape is [batch_size, numFFT, numsymbs, numbits]
    #labels shape is [batch_size, bits of codewords]
    if multi_head:
        loss = 0
        for i in range(logits.shape[0]):
            logits_tmp = tf.transpose(logits[i], perm=[0, 2, 1, 3]) #(batchsize, numSymbol, numFFT, numbits)
            if (logits_tmp.shape[1]*logits_tmp.shape[2]*logits_tmp.shape[3] - labels.shape[1])//(logits_tmp.shape[2]*logits_tmp.shape[3])==2:
                dmrs_loc = {2,11}
            else:
                dmrs_loc = {2}
            #print(dmrs_loc)
            valid_indices = [i for i in range(14) if i not in dmrs_loc]#delete dmrs
            logits_tmp = tf.gather(logits_tmp,valid_indices,axis=1)
            batch_size = logits_tmp.shape[0]
            logits_tmp = tf.reshape(logits_tmp,[batch_size, -1])
            bce = function(labels,logits_tmp)
            rate = tf.constant(1.0, tf.float32) - bce / tf.math.log(2.)
            loss += -1*rate
        loss = loss/(i+1)
    else:
        logits = tf.transpose(logits, perm=[0, 2, 1, 3]) #(batchsize, numSymbol, numFFT, numbits)
        #print(labels.shape)
        #print(logits.shape)
        if (logits.shape[1]*logits.shape[2]*logits.shape[3] - labels.shape[1])//(logits.shape[2]*logits.shape[3])==2:
            dmrs_loc = {2,11}
        else:
            dmrs_loc = {2}
        #print(dmrs_loc)
        valid_indices = [i for i in range(14) if i not in dmrs_loc]#delete dmrs
        logits = tf.gather(logits,valid_indices,axis=1)
        batch_size = logits.shape[0]
        logits = tf.reshape(logits,[batch_size, -1])
        bce = function(labels,logits)
        rate = tf.constant(1.0, tf.float32) - bce / tf.math.log(2.)
        loss = -1*rate
    return loss

def log_metrics(step, loss, rate, summary_writer, mode="train"):
    """记录指标到TensorBoard"""
    with summary_writer.as_default():
        tf.summary.scalar(f"{mode}_loss", loss, step=step)
        tf.summary.scalar(f"{mode}_rate", rate, step=step)

def train_model(model_weights_path):
    epochs = 100
    training_batch_size = 10
    training_logdir = "train_log"
    label = '64QAM-cnn-recevier'
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(training_logdir, f"{label}-{current_time}")
    summary_writer = tf.summary.create_file_writer(logdir)
    model = NNrecevierModel(num_bits_per_symbol=6, cgnn_flag=False, training=True)
    dummyData = tf.random.uniform([1, 128, 14, 4]) # Shape: [batch_size, n_subcarrier, num_symbols, num_rx_ant]
    model(dummyData)
    model.summary()
    # Check if model weights exist
    if os.path.exists(model_weights_path):
        print('load exist weights,and training continue')
        with open(model_weights_path, 'rb') as f:
            weights = pickle.load(f)
        for i, w in enumerate(weights):
            model.nnrecevier.weights[i].assign(w)

    print('Generating data online...')
    # Create OFDM configuration matching the model's expected input shape
    # Assuming data_generator produces received frequency-domain symbols and original bits
    cfg = load_config(r'config.yaml')  # Load OFDM configuration
    # Define steps per epoch for online generation
    steps_per_epoch = 10 # Example value, can be adjusted

    # Create TensorFlow dataset from online generator
    # The generator is assumed to yield (received_freq_symbols, original_bits)
    # received_freq_symbols shape: (batch_size, num_rx_ant, num_symbols, n_subcarrier)
    # original_bits shape: (batch_size, k)
    dataset = create_tf_dataset(cfg, training_batch_size)

    global_steps = epochs * steps_per_epoch
    dataset = dataset.repeat()
    optimizer = tf.keras.optimizers.AdamW(learning_rate=ThreePhaseLR(target_lr=0.001, total_steps=global_steps, warmup_steps=int(global_steps*0.02),
                                                                     decay_start=int(0.1*global_steps)), weight_decay=1e-4, clipnorm=2.)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step,(train_data, llr_label) in enumerate(dataset.take(steps_per_epoch)):
            # Transpose train_data to match model expected input shape [batch_size, n_subcarrier, num_symbols, num_rx_ant]
            # Current shape from generator: [batch_size, num_rx_ant, num_symbols, n_subcarrier]
            train_data = tf.transpose(train_data, perm=[0, 3, 2, 1])
            # Forward pass
            global_step = epoch * steps_per_epoch + step
            with tf.GradientTape() as tape:
                llr_logits = model(train_data)
                loss = loss_function(llr_logits,llr_label,BCE)
            # Computing and applying gradients
            weights = tape.watched_variables()
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            log_metrics(global_step, loss, -1*loss, summary_writer, mode="train")
            print('Iteration {}/{},LR:{:.6}  Rate: {:.4f} bit'.format(global_step + 1, global_steps, optimizer.learning_rate.numpy(), -1*loss.numpy()), end='\n')
            if (global_step + 1) % 500 == 0 or (global_step + 1) == global_steps:
                # Save the weights in a file
                weights = model.nnrecevier.weights
                model_weights_path_save = model_weights_path+'epoch{}-step{}'.format(epoch,global_step)
                with open(model_weights_path_save, 'wb') as f:
                    pickle.dump(weights, f)
    print("\n" + "-"*50)  # 分隔线


def transferOnnx(model_weights_path = 'weights-GroupNormNerualRecevier-matlab-data-train'):
    # export onnx model
    deepRx2 = NNrecevierModel(num_bits_per_symbol=8, cgnn_flag=True, multi_head=True, training=False)
    dummyData = tf.random.uniform([1, 128, 14, 10])
    deepRx2(dummyData)
    deepRx2.summary()
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    for i, w in enumerate(weights):
        deepRx2.nnrecevier.weights[i].assign(w)
    deepRx2.trainable = False
    num_it = [8,4,2,1]
    for it in num_it:
        deepRx2.nnrecevier.num_it = it
        output_path = model_weights_path+f'it{it}.onnx'
        input_signature = [tf.TensorSpec(shape=[None, None, None, 10], dtype=tf.float32, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(
            deepRx2,
            input_signature=input_signature,
            opset=18,
            output_path=output_path)
        print("ONNX 模型已导出到 {}".format(output_path))
        # 加载 ONNX 模型
        onnx_model = onnx.load(output_path)
        # 检查模型是否有效
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证成功")

# 使用示例
if __name__ == "__main__":
    # model = NNrecevierModel(8, True)
    # dummydata = tf.random.uniform([1, 128, 14, 8])
    # model(dummydata)
    # model.summary()
    # dummydata = tf.random.uniform([1, 312, 12, 8])
    # out=model(dummydata)
    # print(out.shape)

    model_weights_path = r'weights\64QAM-test'
    train_model(model_weights_path)
    # transferOnnx(model_weights_path)
