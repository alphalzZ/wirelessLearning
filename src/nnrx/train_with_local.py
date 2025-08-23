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


import onnxruntime as ort
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
from src.ofdm_rx import (
    remove_cp_and_fft,
    estimate_timing_offset,
    estimate_frequency_offset,
    compensate_frequency_offset,
    compensate_timing_offset,
    estimate_channel
    )

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
    @tf.function()
    def call(self, inputs):
        return self.nnrecevier(inputs)

def run_onnx(inputData, onnxModelPath, sess = None):
    
    #only support cdmgroupwithoutdata = 2,Rx = 2
    if not sess:
        sess = ort.InferenceSession(onnxModelPath)
    # 获取输入信息
    input_name = sess.get_inputs()[0].name
    # 生成随机测试数据 (根据实际需求修改)
    inputData = np.array(inputData)
    inputData = inputData.astype(np.float32)
    outputs = sess.run(None, {input_name: inputData})
    # LLRout = process_3d_array(np.squeeze(outputs[0]), dmrsLoc)
    return outputs[0], sess

def loss_function(logits, labels, function, cfg:OFDMConfig, multi_head=False):
    #logits shape is [batch_size, numFFT, numsymbs, numbits]
    #labels shape is [batch_size, bits of codewords]
    if multi_head:
        loss = 0
        for i in range(logits.shape[0]):
            logits_tmp = tf.transpose(logits[i], perm=[0, 2, 1, 3]) #(batchsize, numSymbol, numFFT, numbits)
            # if (logits_tmp.shape[1]*logits_tmp.shape[2]*logits_tmp.shape[3] - labels.shape[1])//(logits_tmp.shape[2]*logits_tmp.shape[3])==2:
            #     dmrs_loc = {2,11}
            # else:
            #     dmrs_loc = {2}
            # #print(dmrs_loc)
            # valid_indices = [i for i in range(14) if i not in dmrs_loc]#delete dmrs
            valid_indices = cfg.get_data_symbol_indices()
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
        # if (logits.shape[1]*logits.shape[2]*logits.shape[3] - labels.shape[1])//(logits.shape[2]*logits.shape[3])==2:
        #     dmrs_loc = {2,11}
        # else:
        #     dmrs_loc = {2}
        # #print(dmrs_loc)
        # valid_indices = [i for i in range(14) if i not in dmrs_loc]#delete dmrs
        valid_indices = cfg.get_data_symbol_indices()
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
    epochs = 1000
    training_batch_size = 20
    training_logdir = "train_log"
    label = '64QAM-cnn-recevier'
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(training_logdir, f"{label}-{current_time}")
    summary_writer = tf.summary.create_file_writer(logdir)
    model = NNrecevierModel(num_bits_per_symbol=6, cgnn_flag=False, training=True)
    dummyData = tf.random.uniform([1, 128, 14, 8]) # Shape: [batch_size, n_subcarrier, num_symbols, 2*num_rx_ant]
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
<<<<<<< HEAD
    steps_per_epoch = 5 # Example value, can be adjusted

    global_steps = epochs * steps_per_epoch
=======
    steps_per_epoch = 10 # Example value, can be adjusted
    global_steps = epochs * steps_per_epoch

>>>>>>> 8227443ca052392c62e07c847de5939f2bd23955
    optimizer = tf.keras.optimizers.AdamW(learning_rate=ThreePhaseLR(target_lr=0.001, total_steps=global_steps, warmup_steps=int(global_steps*0.02),
                                                                     decay_start=int(0.1*global_steps)), weight_decay=1e-4, clipnorm=2.)
    for epoch in range(epochs):
        # Create TensorFlow dataset from online generator
        # The generator is assumed to yield (received_freq_symbols, original_bits)
        # received_freq_symbols shape: (batch_size, num_rx_ant, num_symbols, n_subcarrier)
        # original_bits shape: (batch_size, k)
<<<<<<< HEAD
        cfg.snr_db = np.random.randint(10, 25)  # Random SNR for each epoch
        cfg.timing_offset = np.random.randint(0, 20)  # Random timing offset for each epoch
        cfg.freq_offset = np.random.uniform(-0.05, 0.05)  # Random frequency offset for each epoch
        dataset = create_tf_dataset(cfg, training_batch_size)
        dataset = dataset.repeat()
        
=======
        cfg.snr_db = np.random.randint(5,30)
        cfg.freq_offset = np.random.uniform(-0.05,0.05)
        cfg.timing_offset = np.random.randint(0,100)
        dataset = create_tf_dataset(cfg, training_batch_size)    
        dataset = dataset.repeat()    
>>>>>>> 8227443ca052392c62e07c847de5939f2bd23955
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step,(train_data, llr_label) in enumerate(dataset.take(steps_per_epoch)):
            # Transpose train_data to match model expected input shape [batch_size, n_subcarrier, num_symbols, num_rx_ant]
            # Current shape from generator: [batch_size, num_rx_ant, num_symbols, n_subcarrier]
            train_data = tf.transpose(train_data, perm=[0, 3, 2, 1])
            # Forward pass
            global_step = epoch * steps_per_epoch + step
            with tf.GradientTape() as tape:
                llr_logits = model(train_data)
                loss = loss_function(llr_logits, llr_label, BCE, cfg)
            # Computing and applying gradients
            weights = tape.watched_variables()
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            log_metrics(global_step, loss, -1*loss, summary_writer, mode="train")
            print('Iteration {}/{},LR:{:.6}  Rate: {:.4f} bit'.format(global_step + 1, global_steps, optimizer.learning_rate.numpy(), -1*loss.numpy()), end='\n')
            if (global_step + 1) % 500 == 0 or (global_step + 1) == global_steps:
                # Save the weights in a file
                weights = model.nnrecevier.weights
                model_weights_path_save = model_weights_path+'-epoch{}-step{}'.format(epoch,global_step)
                with open(model_weights_path_save, 'wb') as f:
                    pickle.dump(weights, f)
    print("\n" + "-"*50)  # 分隔线

def ofdm_nnrx(model_weights_path, signal:np.ndarray, cfg:OFDMConfig, run_onnx_flag=True):
    
    if signal.ndim == 1:
        signal = signal[None, :]

    num_ant = signal.shape[0]

    # 1. 移除循环前缀并进行FFT
    rx_symbols = remove_cp_and_fft(signal, cfg)
    
    # 2. 使用导频进行频偏估计和补偿
    offset = cfg.get_subcarrier_offset()
    pilot_symbol_indices = cfg.get_pilot_symbol_indices()
    pilot_symbols = cfg.get_pilot_symbols(pilot_symbol_indices)#排列顺序为第一根天线第一个DMRS，第二根天线，第一个DMRS，第一根天线，第二个DMRS，...
    pilot_symbols = pilot_symbols.reshape(cfg.num_tx_ant,len(pilot_symbol_indices), -1)  # (num_tx_ant, num_pilots, n_pilots)
    pilot_indices = cfg.get_pilot_indices() - offset
    est_timing = []
    est_freq_offset = []
    for a in range(num_ant):
        est_timing.append(
            estimate_timing_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
        )
        est_freq_offset.append(
            estimate_frequency_offset(rx_symbols[a], pilot_symbols, pilot_indices, cfg)
        )
    if cfg.num_tx_ant == 1:
        est_time = np.array(est_timing)[:,None]
        est_freq = np.array(est_freq_offset)[:,None]
    else:
        est_time = np.array(est_timing)
        est_freq = np.array(est_freq_offset)
    est_timing = np.mean(est_time,axis=1)
    est_freq_offset = np.mean(est_freq, axis=1)

    # 逐层逐天线补偿频偏
    comp_ant = [
        compensate_frequency_offset(
            signal[a],
            est_freq_offset[a],
            cfg,
        )
        for a in range(num_ant)
    ]
    signal_freq_comp = np.stack(comp_ant, axis=0)

    # CP 移除与FFT
    rx_symbols = remove_cp_and_fft(signal_freq_comp, cfg)

    # 逐层逐天线补偿时延
    comp_ant = [
        compensate_timing_offset(
            rx_symbols[a],
            est_timing[a],
            cfg,
        )
        for a in range(num_ant)
    ]
    signal_timing = np.stack(comp_ant, axis=0) # (num_ant, num_symbols, n_subcarrier)
    # 3. 信道估计和均衡
    num_layer = cfg.num_tx_ant
    h_est_ant = []
    for a in range(num_ant):
        h_est_tmp = estimate_channel(signal_timing[a], cfg)
        h_est_ant.append(h_est_tmp)
    h_est = np.stack(h_est_ant, axis=0)
    rx_real_part = np.real(signal_timing)
    rx_imag_part = np.imag(signal_timing)
    h_real_part = np.real(h_est)
    h_imag_part = np.imag(h_est)    
    input_data = np.concatenate((rx_real_part, rx_imag_part,
                                    h_real_part, h_imag_part), axis=0, dtype=np.float32)  # 将实部和虚部堆叠成最后一维
    input_data = tf.transpose(input_data[None,...], perm=[0, 3, 2, 1])
    if not run_onnx_flag:
        model = NNrecevierModel(num_bits_per_symbol=6, cgnn_flag=False, training=False)
        dummyData = tf.random.uniform([1, 128, 14, 8]) # Shape: [batch_size, n_subcarrier, num_symbols, 2*num_rx_ant]
        model(dummyData)
        # Check if model weights exist
        if os.path.exists(model_weights_path):
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            for i, w in enumerate(weights):
                model.nnrecevier.weights[i].assign(w)
        else:
            raise ValueError("model_weights_path 不存在！！！")    
        llr = model(input_data) #llr size is [batch, n_subcarriers, n_symbols, n_bits_per_symbol]
    else:
        input_data = input_data.numpy()
        llr, sess = run_onnx(input_data, model_weights_path+'.onnx', cfg.sess)
        if  cfg.sess is None:
            cfg.sess = sess
    llr = tf.gather(llr, cfg.get_data_symbol_indices(), axis=2)
    llr = tf.transpose(llr, perm=[0, 2, 1, 3]).numpy() #(batchsize, n_symbols, n_subcarriers, n_bits_per_symbol)
    llr = llr[...,:cfg.mod_order].reshape(-1)
    bits = (llr > 0).astype(np.int8)
    return bits

def ofdm_nnrx_matlab(rx_symbols_real, rx_symbols_imag, pilot_symbol_indices, pilot_symbols_real, pilot_symbols_imag, pilot_indices,nfft,mod_order,plot=False):
    # simulation with matlab
    run_onnx_flag = True
    rx_symbols = rx_symbols_real+1j*rx_symbols_imag
    pilot_symbols = pilot_symbols_real+1j*pilot_symbols_imag
    num_tx_ant = pilot_symbols.shape[0]
    pilot_indices = pilot_indices.astype(np.int32)
    dummyCfg = OFDMConfig()
    dummyCfg.num_tx_ant = num_tx_ant
    dummyCfg.pilot_symbols = pilot_symbol_indices.astype(np.int32)
    dummyCfg.n_fft = nfft
    dummyCfg.pilot_spacing = pilot_indices[1] - pilot_indices[0]
    dummyCfg.n_subcarrier = rx_symbols.shape[2]
    dummyCfg.num_rx_ant = rx_symbols.shape[0]
    dummyCfg.interp_method = 'linear'
    dummyCfg.mod_order = np.int8(mod_order)
    dummyCfg.num_symbols = rx_symbols.shape[1]
    dummyCfg.equ_method = 'mmse'
    dummyCfg.win_size = [2,2,2]
    num_layer = dummyCfg.num_tx_ant
    num_ant = dummyCfg.num_rx_ant

    est_timing = []
    est_freq_offset = []
    est_timing = []
    est_freq_offset = []
    for a in range(num_ant):
        est_timing.append(
            estimate_timing_offset(rx_symbols[a], pilot_symbols, pilot_indices, dummyCfg)
        )
        est_freq_offset.append(
            estimate_frequency_offset(rx_symbols[a], pilot_symbols, pilot_indices, dummyCfg)
        )
    if dummyCfg.num_tx_ant == 1:
        est_time = np.array(est_timing)[:,None]
        est_freq = np.array(est_freq_offset)[:,None]
    else:
        est_time = np.array(est_timing)
        est_freq = np.array(est_freq_offset)
    est_timing = np.mean(est_time,axis=1)
    est_freq_offset = np.mean(est_freq, axis=1)
    if dummyCfg.display_est_result:
        print(f"估计的时延: {est_timing}, 估计的频偏: {est_freq_offset}")

    rx_symbols = np.stack(
        [
            compensate_timing_offset(rx_symbols[a], est_timing[a], dummyCfg)
            for a in range(dummyCfg.num_rx_ant)
        ],
        axis=0
    )

    rx_symbols_freq_compensation = np.stack(
        [
            compensate_frequency_offset(rx_symbols[a], est_freq_offset[a], dummyCfg)
            for a in range(dummyCfg.num_rx_ant)
        ],
        axis=0,
    )
    #channel est
    # 3. 信道估计和均衡
    h_est_ant = []
    for a in range(num_ant):
        h_est_tmp = estimate_channel(rx_symbols_freq_compensation[a], dummyCfg, pilot_symbols[0], pilot_indices)
        h_est_ant.append(h_est_tmp)
    h_est = np.stack(h_est_ant, axis=0)
    rx_real_part = np.real(rx_symbols_freq_compensation)
    rx_imag_part = np.imag(rx_symbols_freq_compensation)
    h_real_part = np.real(h_est)
    h_imag_part = np.imag(h_est)    
    input_data = np.concatenate((rx_real_part, rx_imag_part,
                                    h_real_part, h_imag_part), axis=0, dtype=np.float32)  # 将实部和虚部堆叠成最后一维
    input_data = tf.transpose(input_data[None,...], perm=[0, 3, 2, 1])
    if not run_onnx_flag:
        model = NNrecevierModel(num_bits_per_symbol=6, cgnn_flag=False, training=False)
        dummyData = tf.random.uniform([1, 128, 14, 8]) # Shape: [batch_size, n_subcarrier, num_symbols, 2*num_rx_ant]
        model(dummyData)
        # Check if model weights exist
        if os.path.exists(model_weights_path):
            with open(model_weights_path, 'rb') as f:
                weights = pickle.load(f)
            for i, w in enumerate(weights):
                model.nnrecevier.weights[i].assign(w)
        else:
            raise ValueError("model_weights_path 不存在！！！")    
        llr = model(input_data) #llr size is [batch, n_subcarriers, n_symbols, n_bits_per_symbol]
    else:
        model_weights_path = r'./weights/64QAM-testepoch999-step9999-epoch999-step9999'
        input_data = input_data.numpy()
        llr,_ = run_onnx(input_data, model_weights_path+'.onnx')
    llr = tf.gather(llr, dummyCfg.get_data_symbol_indices(), axis=2)
    llr = tf.transpose(llr, perm=[0, 2, 1, 3]).numpy() #(batchsize, n_symbols, n_subcarriers, n_bits_per_symbol)
    llr = llr[...,:dummyCfg.mod_order].reshape(-1)

    return -1*llr

def transferOnnx(model_weights_path = 'weights-GroupNormNerualRecevier-matlab-data-train'):
    import tf2onnx
    import onnx
    # export onnx model
    deepRx2 = NNrecevierModel(num_bits_per_symbol=6, cgnn_flag=False, multi_head=False, training=False)
    dummyData = tf.random.uniform([1, 128, 14, 8])
    deepRx2(dummyData)
    deepRx2.summary()
    with open(model_weights_path, 'rb') as f:
        weights = pickle.load(f)
    for i, w in enumerate(weights):
        deepRx2.nnrecevier.weights[i].assign(w)
    deepRx2.trainable = False
    output_path = model_weights_path+f'.onnx'
    input_signature = [tf.TensorSpec(shape=[None, None, None, 8], dtype=tf.float32, name='input')]
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

    model_weights_path = r'D:\pyHome\projs\wirelessLearning-support-2-layer\weights\64QAM-testepoch999-step9999-epoch999-step9999'
    # train_model(model_weights_path)
    transferOnnx(model_weights_path)
