# transformer_ofdm_schemeC_demo.py
# -*- coding: utf-8 -*-
import os
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard.writer import SummaryWriter

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.TransformerRx.data_generator_torch import create_pytorch_dataloader,rb_view_reverse
from src.TransformerRx.basic_modules import *
from src.TransformerRx.transformer_ofdm_schemeC_plus_demo import *
from src.config import OFDMConfig


Number = Union[int, float]


class TBLogger:
    """
    轻量封装的 TensorBoard 日志器：
      - 自动拼接时间戳：log_name_YYYYmmdd-HHMMSS
      - log(dict) 一次性记录多指标
      - 可选 prefix，便于分类（如 'train/'、'val/'）
      - 简单、易插拔：log() / close()
    """
    def __init__(self, log_name: str, base_dir: str = "runs") -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{log_name}_{ts}"
        self.log_dir = os.path.join(base_dir, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(
        self,
        metrics: Dict[str, Union[Number, torch.Tensor]],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """
        metrics: {'Loss/train': 0.1, 'Acc/val': 0.92, ...}
        step   : 训练步数或 epoch 索引；不传则让 SummaryWriter 自增
        prefix : 给所有 tag 加统一前缀（如 'train/'）
        """
        pre = f"{prefix}" if (prefix == "" or prefix.endswith("/")) else f"{prefix}/"
        for k, v in metrics.items():
            # 安全转换为 float
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.detach().float().cpu().item()
                else:
                    v = float(v.detach().float().mean().cpu().item())
            else:
                v = float(v)
            self.writer.add_scalar(pre + k, v, global_step=step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        self.writer.add_text(tag, text, global_step=step)

    def add_figure(self, tag: str, figure, step: Optional[int] = None) -> None:
        self.writer.add_figure(tag, figure, global_step=step, close=True)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def create_tb_logger(log_name: str, base_dir: str = "runs") -> TBLogger:
    """工厂函数：更贴近“灵活插拔”的使用心智。"""
    return TBLogger(log_name=log_name, base_dir=base_dir)

def lr_schedule(total_steps,step, warmup, lr_max, lr_min):
    #lr_max,lr_min = 1e-3, 1e-8
    if step < warmup:   # linear warmup
        return (step+1) / warmup
    # cosine to lr_min
    prog = (step - warmup) / max(1, total_steps - warmup)
    cos = 0.5*(1 + math.cos(math.pi*prog))
    return cos * (1 - lr_min/lr_max) + (lr_min/lr_max)

# ----------------------------
# 简单可运行演示（dummy数据）
# ----------------------------
def demo_run_basic():
    torch.manual_seed(0)
    n_sym=14
    # 配置
    model_cfg = ModelCConfig(
        d_in=8,
        d_model=64,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=6,
        dim_ff=128,
        dropout=0.1,
        bits_per_symbol=6,  # 16QAM
    )

    BATCH_SIZE = 4
    EPOCHS = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelC(model_cfg, n_sym=n_sym).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    if 0:
        model_path = r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_stage2_bce.pth'
        load_full(model,model_path,device)
    # # 假设 B=16（两帧），每帧 16 个 RB -> B3 = B*3 = 6
    # B = 16
    # B3 = B * 16
    # d_in = cfg.d_in

    # # 构造 dummy 特征：
    # # - enc_feats: [B3, 24, d_in]，可类比 "LS通道的 Re/Im/|·|/angle" 等
    # # - dec_feats: [B3, 144, d_in]，可类比 "接收符号的 Re/Im/|·|/angle" 等
    # enc_feats = torch.randn(B3, 24, d_in, device=device)
    # dec_feats = torch.randn(B3, 144, d_in, device=device)
    clipper = ClipStdPerBit(Lmax_per_bit=[6,6,6,6,6,6])
    for epoch in range(EPOCHS):
        n_subcarrier = np.random.choice([168, 192, 180, 204, 216, 480,240,120])
        snr_db = np.random.uniform(15, 40)  
        print("epoch:{},snr_db:{}".format(epoch, snr_db))
        ofdm_cfg = OFDMConfig(
            n_fft=2048,
            n_subcarrier = n_subcarrier,
            cp_len=144,
            mod_order=6,  #64QAM
            num_symbols=n_sym,
            num_tx_ant=1,
            num_rx_ant=2,
            snr_db=snr_db,
            channel_type='multipath'
        )
        train_set = create_pytorch_dataloader(ofdm_cfg, BATCH_SIZE)
        for i,(x_pilot_batch,x_data_batch, y_batch, z_batch) in enumerate(train_set):
            clipper.fit(z_batch)
            if i>20:
                break
        for i,(x_pilot_batch,x_data_batch, y_batch, z_batch) in enumerate(train_set):
            # 前向
            llr_logits = model(x_pilot_batch, x_data_batch)  # [B3, 144, bits]
            llr_logits = rb_view_reverse(llr_logits, BATCH_SIZE)
            # 伪造 label（随机bit），做一轮 BCE 训练步
            loss_bce = F.binary_cross_entropy_with_logits(llr_logits, y_batch) #（y_batch, tx_bits)
            rate = 1.-loss_bce/0.69314718
            z_batch = clipper.forward(z_batch)
            loss_mse = F.mse_loss(llr_logits, -1*z_batch) #TODO max norm z_batch（llr）
            loss_total = loss_bce+0.1*loss_mse
            print("step:{},BCE loss:{}, MSE loss:{}, total lass:{},  rate: {}".format(i+1, float(loss_bce.item()),
                     float(loss_mse.item()), float(loss_total.item()), float(rate.item())))
            
            # 一步优化演示
            optim.zero_grad()
            loss_bce.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if i > 50:
                break
    save_full(model,r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_stage2_bce.pth')

def demo_run_plus():

    continue_train = 1
    retrain_hrefine = 1
    torch.manual_seed(0)
    n_sym=14
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelCPlusConfig(
        d_in_pilot=10, d_in_data=8, d_model=128, d_model_data=128, nhead=8, nhead_data=8,
        num_enc_layers=4, num_dec_layers=8,
        dim_ff=512,dim_ff_data=512, dropout=0.1, bits_per_symbol=6
    )
    model = ModelCPlus(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    BATCH_SIZE = 8
    EPOCHS = 5000
    
    total_steps = 1 * EPOCHS
    scheduler = LambdaLR(optim, lr_lambda=lambda step:lr_schedule(total_steps, step, 0,1e-4, 1e-8))
    if continue_train:
        model_path = r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_base_stage1_bce.pth'
        load_full(model,model_path,device)
    else:
        # 加载保存的hrefine参数
        h_refine_state_dict = torch.load(r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\h_refine_weights.pth')
        # 将参数加载到模型的 HRefine1D 部分
        model.h_refine.load_state_dict(h_refine_state_dict)
        print("HRefine1D 的参数已加载到模型中")
        
    if not retrain_hrefine:
        # 冻结 HRefine1D 的权重
        for param in model.h_refine.parameters():
            param.requires_grad = False
    # # 假设 B=16（两帧），每帧 16 个 RB -> B3 = B*3 = 6
    # B = 16
    # B3 = B * 16
    # d_in = cfg.d_in

    # # 构造 dummy 特征：
    # # - enc_feats: [B3, 24, d_in]，可类比 "LS通道的 Re/Im/|·|/angle" 等
    # # - dec_feats: [B3, 144, d_in]，可类比 "接收符号的 Re/Im/|·|/angle" 等
    # enc_feats = torch.randn(B3, 24, d_in, device=device)
    # dec_feats = torch.randn(B3, 144, d_in, device=device)
    global_step = 0
    tb = create_tb_logger("transformer", base_dir=r"D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\log")
    for epoch in range(EPOCHS):
        n_subcarrier = np.random.choice([216])
        snr_db = np.random.uniform(0, 38)  
        channel_type = np.random.choice(["multipath"])
        if channel_type == "awgn":
            snr_db = np.random.uniform(15, 25)  
        #print("epoch:{},snr_db:{},channel_type:{}".format(epoch, snr_db, channel_type))
        ofdm_cfg = OFDMConfig(
            n_fft=1024,
            n_subcarrier = n_subcarrier,
            cp_len=144,
            mod_order=6,  #64QAM
            num_symbols=n_sym,
            num_tx_ant=1,
            num_rx_ant=2,
            snr_db=snr_db,
            channel_type=channel_type
        )
        train_set = create_pytorch_dataloader(ofdm_cfg, BATCH_SIZE)
        for i,(x_pilot_batch,x_data_batch, y_batch, z_batch) in enumerate(train_set):
            # 前向
            if i > 0:
                break
            out = model(x_pilot_batch, x_data_batch)  # [B3, 144, bits]
            llr_logits = rb_view_reverse(out["logits"], BATCH_SIZE)
            mse_loss = out["mse_loss"]
            mes_loss_hat = out["loss_h_est"]
            #print("llr example:", llr_logits[:,:10])
            # 伪造 label（随机bit），做一轮 BCE 训练步
            loss_bce = F.binary_cross_entropy_with_logits(llr_logits, y_batch) #（y_batch, tx_bits)
            loss_total = 0.*mse_loss + 1.0*loss_bce
            rate = 1.-loss_bce/0.69314718
            print("epoch:{},step:{},loss_total:{}, loss_mse:{}, mes_loss_hat:{}, loss_bce:{}, rate: {}, lr: {}".format(epoch, i+1, float(loss_total.item()), float(mse_loss.item()), float(mes_loss_hat.item()),
                float(loss_bce.item()), float(rate.item()),optim.param_groups[0]['lr']))
            
            # 一步优化演示
            optim.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            global_step += 1
            metrics = {"loss": loss_total.item(), "rate": rate.item(), "lr": optim.param_groups[0]['lr']}
            tb.log(metrics, step=global_step, prefix="train")
        #if (epoch+1)%100 == 0 and (epoch+1) != EPOCHS:
        #   save_path = r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_plus_Asymmetric_stage3_focal_epoch{}.pth'.format(epoch+1)
        #   save_full(model, save_path)
    save_full(model,r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_base_stage1_bce.pth')
    #h_refine_state_dict = model.h_refine.state_dict()  # 提取 HRefine1D 的参数
    #torch.save(h_refine_state_dict, r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\h_refine_weights.pth')
    #print("HRefine1D 的参数已保存到 D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\h_refine_weights.pth")

def save_h_refine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelCPlusConfig(
        d_in_pilot=10, d_in_data=8, d_model=128, d_model_data=128, nhead=8, nhead_data=8,
        num_enc_layers=4, num_dec_layers=8,
        dim_ff=512,dim_ff_data=512, dropout=0.1, bits_per_symbol=6
    )
    model = ModelCPlus(cfg).to(device)
    model_path = r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\transformer_recevier_base_stage1_bce.pth'
    load_full(model,model_path,device)
    h_refine_state_dict = model.h_refine.state_dict()  # 提取 HRefine1D 的参数
    torch.save(h_refine_state_dict, r'D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\h_refine_weights.pth')
    print("HRefine1D 的参数已保存到 D:\pyHome\projs\wirelessLearning-support-2-layer\src\TransformerRx\weights\h_refine_weights.pth")
    
    

if __name__ == "__main__":
    demo_run_plus()
    #save_h_refine()
