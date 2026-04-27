"""
### Summary of modification

1. Loss Function: created a new loss `Mask_DC_and_CE_loss` for multi-class
2. Training Script: use `CustomTask` configuration
3. Evaluation: modified `eval_mask_slice2` to correctly compute the average Dice score across all classes
4. Model Architecture: `get_model(args.modelname, ...)` in `train_video.py`) is configured to output `C` channels for `C` classes.

CUDA_VISIBLE_DEVICES=1 nohup \
    python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/MemSAM/train_video_multi_class.py \
    --task Task2 --modelname SAM \
    > /mnt/hdd2/task2/memsam/train_200.log 2>&1 &

"""

#==================================================================================================================================================#
"""
2026年04月15+16+17号
注意 每次五折训练前 必须修改config.py中的train_split, val_split
目前对于本代码文件 作出修改如下
    1. 早停：lr初始固定在1e-4 连续5个patience就会降低lr到1e-5 如果还不能提升 就会停止训练
    2. 加上了全面的评估指标 iou/dice/hd95 在evulation.py同步更新了
    3. data_us.py EchoVideoDataset 修改完毕
    4. 训练结束后出图
    5. 新增一个fold参数
"""
#=======================================================Debug出现的问题以及修复方法=====================================================================#
"""
训练Debug如下：
    1. 原来的训练代码 config.py里面用的256px训练 (img_size) 但是SAM里面的位置编码是1024 -> 在config.py里面统一分辨率是1024*1024
    2. 多分类问题 原生SAM的mask decoder只支持二分类 -> 在mask_decoder.py加了一个1*1的卷积层 修改了model_dict.py的初始化逻辑 可以根据生成的class.json里面的类别数量来初始化mask decoder
    3. 维度问题 数据集是视频序列 维度是(B, T, C, H, W) 但是SAM期望的是2d图像 也就是四维的 不是五维的 -> 在本代码文件和evaluation.py加了一个维度压缩的步骤
       具体做法是 训练前将将[B, T]压平为单个batch维度传给模型 然后训练后将输出重新reshape回[B, T, C, H, W]进行loss计算和指标评估
       和我们的SAM_LRU有一点点像
    4. 有个报错是Indexing Assertion Error -> 数据集里面class.json是30个class 但是config.py里面是29个 所以会索引越界 现在改成30了
    5. 因为我们需要hd95 原代码好像没有用到 然后evaluation计算时因为数据类型导致计算异常 -> 在evaluation.py对Hausdorff的输入转换成float32
    6. 新增一个fold参数

用例(fold0):
    CUDA_VISIBLE_DEVICES=1 nohup \
    python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/MemSAM/train_video_multi_class.py \
    --task Task2 --modelname SAM \
    > /mnt/hdd2/task2/memsam/train_fold0.log 2>&1 &

切换fold前注意修改config.py!!!!

结果保存在/mnt/hdd2/task2/memsam/checkpoints/Task2/

CUDA_VISIBLE_DEVICES=0 nohup \
python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/MemSAM/train_video_multi_class.py \
    --task Task2 --modelname SAM \
    --encoder_input_size 1024 --low_image_size 256 --frame_length 10 \
    --fold 0 --exp_id 1 \
    > /mnt/hdd2/task2/memsam/train_fold0_exp1.log 2>&1 &
"""

#=======================================================后续实验问题=================================================================================#
"""
由于/mnt/hdd2/task2/memsam/train_fold0.log epoch1就best了 后续进行优化
1. 特征维度提升 将cls_head的输入从1通道mask提升到32维图像特征 (这个从exp_id1开始实现)
2. 增加mask_decoder显式解冻
3. 修复iou区间问题
4. 新增exp_id参数
"""

"""
2026年04月17日debug记录

因为换用逻辑：就是把30类问题变成30个二值问题 每次训练输出一个图像+某个类别的bbox 输出一个binary mask然后重复30次

命令：
CUDA_VISIBLE_DEVICES=0 nohup \
python /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/MemSAM/train_video_multi_class.py \
    --task Task2 --modelname MemSAM \
    --encoder_input_size 256 --low_image_size 256 --frame_length 10 \
    --fold 0 --exp_id 2 \
    > /mnt/hdd2/task2/memsam/train_fold0_exp2.log 2>&1 &

修复bug如下：
    1. mask_decoder.py -> 删除cls_head 恢复SAM单通道二值输出
    2. model_dict.py -> MemSAM分支解冻mask_decoder
    3. data_us.py的load_video_and_mask_file未返回frame_inds -> 添加frame_inds返回值
    4. JointTransform3D之前就做了mask二值化 应该先做transform然后再做二值化
    5. BBox坐标尺度不匹配 json里面bbox是1024 但图像resize到256 prompt完全超出图像边界导致预测全背景 -> data_us.py缩放

结论: 
    训练确实可以再次跑通 loss在下降 但是验证集指标一直是0 后来加了debug print 发现如下 具体在/mnt/hdd2/task2/memsam/train_fold0_exp2.log
    类似[Debug Val] clip_idx=40, c_id=9, out range=[-2.665, -2.189], pred_fg_mean=0.0000, gt_fg_mean=0.0026：
        gt有前景(gt_fg_mean大于0的)这个没问题 但是输出out range都是负的 而且不同的c_id和bbox对应的out range基本上一样
        说明bbox prompt对于输出没影响！
        再次看一遍论文 memsam的memory模块是在心脏超声上面预训练的 我们迁移到手术器械或者器官上 偏置太多了
        他们的forward_with_memory只在第一帧使用bbox prompt 记忆模块在后续帧中压过了prompt的引导

所以我觉得我们这个任务+数据集 不适合用memsam做baseline
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
from easydict import EasyDict
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from tqdm import tqdm
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import EchoVideoDataset, JointTransform3D
from utils.data_us import JointTransform2D, EchoDataset
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='XMemSAM', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=256, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='Task2', help='task or dataset name: CAMUS_Video_Full or EchoNet_Video')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='/mnt/hdd2/task2/sam/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', action="store_true", help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--keep_log', action="store_true", help='keep the loss&lr&dice during training or not')
    parser.add_argument('--frame_length', type=int, default=10)
    parser.add_argument('--point_numbers', type=int, default=1)
    parser.add_argument('--enable_memory', action="store_true")
    parser.add_argument('--semi', action="store_true")
    parser.add_argument('--reinforce', action="store_true")
    parser.add_argument('--disable_point_prompt', action="store_true")
    parser.add_argument('--fold', type=int, default=0, help='fold number for cross validation')
    parser.add_argument('--exp_id', type=int, default=0, help='experiment ID')
    args = parser.parse_args()
    print(args)

    # ==================================================parameters setting==================================================

    # override_args = EasyDict(base_lr=0.0001,
    #                     batch_size=1,
    #                     encoder_input_size=256,
    #                     keep_log=True,
    #                     low_image_size=256,
    #                     frame_length=10,
    #                     modelname='XMemSAM',
    #                     n_gpu=1,
    #                     sam_ckpt='checkpoints/sam_vit_b_01ec64.pth',
    #                     task='CAMUS_Video_Full',
    #                     vit_name='vit_b',
    #                     enable_memory=True,
    #                     enable_point_prompt=True,
    #                     point_numbers=1,
    #                     warmup=False,
    #                     warmup_period=250)
    opt = get_config(args.task)
    opt.task = args.task
    opt.semi = args.semi
    
    # REQUIRED: Sync command line args to opt to avoid resolution mismatch between Model and DataLoader
    opt.encoder_input_size = args.encoder_input_size
    opt.low_image_size = args.low_image_size
    opt.img_size = args.encoder_input_size
    opt.batch_size = args.batch_size
    opt.modelname = args.modelname

    # Append fold and exp_id to output paths for better organization
    fold_str = f'fold{args.fold}_exp_id{args.exp_id}'
    opt.save_path = os.path.join(opt.save_path, fold_str)
    opt.result_path = os.path.join(opt.result_path, fold_str)
    opt.tensorboard_path = os.path.join(opt.tensorboard_path, fold_str)
    
    # Ensure trailing slashes for the '+' concatenations used elsewhere in the script
    if not opt.save_path.endswith('/'): opt.save_path += '/'
    if not opt.result_path.endswith('/'): opt.result_path += '/'
    if not opt.tensorboard_path.endswith('/'): opt.tensorboard_path += '/'

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime(
            '%m%d%H%M'
        )  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    # ==================================================set random seed==================================================
    seed_value = 301  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) 

    # ==================================================build model==================================================
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform3D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # tf_train, tf_val = None, None

    train_split_path = os.path.join(opt.data_path, opt.train_split)
    val_split_path = os.path.join(opt.data_path, opt.val_split)

    train_dataset = EchoVideoDataset(opt.data_path, train_split_path, tf_train, img_size=args.encoder_input_size,frame_length=args.frame_length, point_numbers=args.point_numbers, disable_point_prompt=args.disable_point_prompt)
    val_dataset = EchoVideoDataset(opt.data_path, val_split_path, tf_val, img_size=args.encoder_input_size,frame_length=args.frame_length, point_numbers=args.point_numbers, disable_point_prompt=args.disable_point_prompt)
    from torch.utils.data.dataloader import default_collate
    def ovr_collate_fn(batch):
        # We need to pop available_classes because it has irregular lengths which default_collate hates
        available_classes = [item.pop('available_classes') for item in batch]
        collated_batch = default_collate(batch)
        collated_batch['available_classes'] = available_classes
        return collated_batch

    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=ovr_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=ovr_collate_fn)


    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    current_lr = 1e-4
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr, betas=(0.9, 0.999), weight_decay=0.1)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    train_loss_history, val_loss_history = [], []
    train_iou_history, val_iou_history = [], []
    val_dice_history, val_hd95_history = [], []

    best_dice = 0.0
    patience_counter = 0
    lr_dropped = 0

    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        train_intersections = 0.0
        train_unions = 0.0
        
        tbar = tqdm(trainloader)
        for batch_idx, (datapack) in enumerate(tbar):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['label'].to(dtype = torch.float32, device=opt.device)
            bboxes = datapack['bbox'].to(dtype = torch.float32, device=opt.device)
            
            optimizer.zero_grad()
            
            # MemSAM binary forward with BBox prompt
            # out shape: (B, T, 1, H, W)
            out = model(imgs, pt=None, bbox=bboxes)
            
            # masks: (B, T, H, W) from DataLoader
            # Convert to (B, T, 1, H, W) to match model output
            target_masks = masks.unsqueeze(2)
            
            loss = criterion(out, target_masks)
            loss.backward()
            optimizer.step()
            
            train_losses += loss.item()
            
            # Metric calculation (Binary IoU)
            with torch.no_grad():
                pred_bin = (out.sigmoid() > 0.5).float()
                inter = torch.sum(pred_bin * target_masks).item()
                union = torch.sum(pred_bin) + torch.sum(target_masks) - inter
                train_intersections += inter
                train_unions += max(union.item() if torch.is_tensor(union) else union, 1e-6)
                
            tbar.set_description(f"Epoch {epoch} Train loss: {loss.item():.4f}")

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        train_loss_hist = train_losses / len(trainloader)
        train_iou_hist = (train_intersections / train_unions) * 100
        print('epoch [{}/{}], train loss:{:.4f}, train IoU:{:.4f}'.format(epoch, opt.epochs, train_loss_hist, train_iou_hist))
        
        train_loss_history.append(train_loss_hist)
        train_iou_history.append(train_iou_hist)
        
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_loss_hist, epoch)
            TensorWriter.add_scalar('learning rate', current_lr, epoch)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            val_iou, val_dice = [], []
            
            print(f"Rigorous evaluation: iterating all classes per clip...")
            vbar = tqdm(valloader)
            for v_idx, datapack in enumerate(vbar):
                imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
                full_masks = datapack['full_mask'].to(device=opt.device) # (B, 1, T, H, W)
                selected_keys = datapack['selected_keys']
                
                # available_classes was returned as a list of integers
                curr_available_classes = datapack['available_classes']
                
                # Since validation batch_size is 1, take the first element (the list of classes for this clip)
                classes_to_test = curr_available_classes[0]
                
                clip_results_iou, clip_results_dice = [], []
                
                for c_id in classes_to_test:
                    c_id = int(c_id)
                    # Use .long() to ensure exact integer comparison for multi-class mask
                    bin_gt = (full_masks.long() == c_id).float()
                    
                    # Fetch BBoxes for THIS class from the dataset's JSON using selected_keys
                    json_bboxes = []
                    for k_name in selected_keys:
                        # k_name may be a string (bs=1) or a list/tuple
                        actual_key = k_name[0] if isinstance(k_name, (list, tuple)) else k_name
                        frame_info = valloader.dataset.bbox_json[actual_key]
                        found_box = [-1, -1, opt.img_size, opt.img_size]
                        for item in frame_info:
                            item_c_id = int(item['mask_path'].split('class')[-1].split('.')[0])
                            if item_c_id == c_id:
                                # Scale BBox from JSON (1024) to model resolution
                                scale = opt.img_size / 1024.0
                                orig_box = item['bbox']
                                found_box = [coord * scale for coord in orig_box]
                                break
                        json_bboxes.append(found_box)
                    
                    curr_bboxes = torch.tensor(json_bboxes, dtype=torch.float32, device=opt.device).unsqueeze(0) # (1, T, 4)
                    
                    with torch.no_grad():
                        # out shape: (B, T, 1, H, W)
                        out = model(imgs, pt=None, bbox=curr_bboxes)
                        # bin_gt: (B, T, H, W) -> unsqueeze to (B, T, 1, H, W)
                        target_masks = bin_gt.unsqueeze(2)
                        
                        # DEBUG PRINTS
                        with torch.no_grad():
                            pred_temp = (out.sigmoid() > 0.5).float()
                            print(f"  [Debug Val] clip_idx={batch_idx}, c_id={c_id}, "
                                  f"out range=[{out.min().item():.3f}, {out.max().item():.3f}], "
                                  f"pred_fg_mean={pred_temp.mean().item():.4f}, "
                                  f"gt_fg_mean={target_masks.mean().item():.4f}")
                        
                        pred = (out.sigmoid() > 0.5).float()
                        
                        inter = torch.sum(pred * target_masks).item()
                        union = torch.sum(pred) + torch.sum(target_masks) - inter
                        
                        clip_results_iou.append(inter / max(union.item() if torch.is_tensor(union) else union, 1e-6))
                        clip_results_dice.append(((2 * inter) / (torch.sum(pred) + torch.sum(target_masks) + 1e-6)).item())
                
                if clip_results_iou:
                    val_iou.append(np.mean(clip_results_iou))
                    val_dice.append(np.mean(clip_results_dice))
                
            mean_iou = np.mean(val_iou) * 100
            mean_dice = np.mean(val_dice) * 100
            
            print('epoch [{}/{}], val dice:{:.4f}, val iou:{:.4f}'.format(epoch, opt.epochs, mean_dice, mean_iou))
            
            val_loss_history.append(0.0) # Placeholder
            val_dice_history.append(mean_dice)
            val_iou_history.append(mean_iou)
            val_hd95_history.append(0.0)
            
            if args.keep_log:
                TensorWriter.add_scalar('val_dice', mean_dice, epoch)
                
            if mean_dice > best_dice:
                best_dice = mean_dice
                patience_counter = 0 
                if not os.path.isdir(opt.save_path): os.makedirs(opt.save_path)
                save_path = f"{opt.save_path}{args.modelname}_best.pth"
                torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
                print(f"==> New best validation Dice: {best_dice:.4f}! Saved.")
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"==> Patience limit reached. Dropping LR.")
                    lr_dropped += 1
                    patience_counter = 0
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1
                    if lr_dropped >= 2:
                        print("==> Early stopping triggered.")
                        break
                        
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if not os.path.isdir(opt.save_path): os.makedirs(opt.save_path)
            torch.save(model.state_dict(), f"{opt.save_path}{args.modelname}_last.pth", _use_new_zipfile_serialization=False)
            
    # ============================ 训练结束：生成CSV和指标曲线 ============================
    print("Training finished! Generating history plots...")
    epochs_range = list(range(len(train_loss_history)))
    
    # Simple Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_iou_history, label='Train IoU')
    if val_iou_history:
        val_epochs = list(range(0, len(val_iou_history) * opt.eval_freq, opt.eval_freq))
        plt.plot(val_epochs, val_iou_history, label='Val IoU')
    plt.title('IoU History')
    plt.legend()
    plt.savefig(os.path.join(opt.save_path, 'history_plots.png'))
    print(f"Plots saved to {opt.save_path}")

    # 填充验证集数据以匹配 train 的维度，方便导出对齐的 CSV
    # 如果 eval_freq 是 1，长度即相等
    metrics_df = pd.DataFrame({
        "Epoch": epochs_range,
        "Train_Mean_Loss": train_loss_history,
        "Train_Mean_IoU": train_iou_history
    })
    
    val_epochs_range = list(range(0, len(val_loss_history) * opt.eval_freq, opt.eval_freq))
    val_df = pd.DataFrame({
        "Epoch": val_epochs_range,
        "Val_Mean_Loss": val_loss_history,
        "Val_Mean_Dice": val_dice_history,
        "Val_Mean_IoU": val_iou_history,
        "Val_Mean_HD95": val_hd95_history
    })
    
    final_df = pd.merge(metrics_df, val_df, on="Epoch", how="left")
    
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    csv_path = os.path.join(opt.save_path, "training_metrics_history.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV to {csv_path}")

    # ===== Requested plotting logic =====
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_loss_history, label='Training Mean Loss', marker='o')
    plt.plot(val_epochs_range, val_loss_history, label='Validation Mean Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Mean Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_iou_history, label='Training Mean IoU', color='orange', marker='o')
    plt.plot(val_epochs_range, val_iou_history, label='Validation Mean IoU', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training and Validation Mean IoU')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_epochs_range, val_dice_history, label='Validation Mean Dice', marker='s')
    plt.plot(val_epochs_range, val_iou_history, label='Validation Mean IoU', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Mean Dice and Mean IoU')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_epochs_range, val_hd95_history, label='Validation Mean HD95', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('HD95')
    plt.title('Validation Mean HD95')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(opt.save_path, "training_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plotting curves to {plot_path}")

if __name__ == '__main__':
    main()