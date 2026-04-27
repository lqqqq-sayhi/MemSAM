import torch
import json
import os
import numpy as np
import traceback
from torch.utils.data import DataLoader
from utils.data_us import EchoVideoDataset
from models.model_dict import get_model

def verify_ovr_setup():
    print("=== 1. Testing Data Loader Alignment ===")
    data_path = '/mnt/hdd2/task2/memsam/dataset_memsam_new/'
    split_path = os.path.join(data_path, 'fold0_train_filenames.txt')
    
    dataset = EchoVideoDataset(
        data_path, 
        split_path, 
        joint_transform=None, 
        img_size=256, 
        frame_length=10
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(loader))
    
    print(f"Batch Image shape: {batch['image'].shape}")
    
    print("\n=== 2. Testing Model Forward Pass (Binary) ===")
    class Args:
        modelname = "MemSAM"
        encoder_input_size = 256
        low_image_size = 256
        frame_length = 10
        point_numbers = 1
        disable_point_prompt = True
        classes = 2 
        sam_ckpt = "/mnt/hdd2/task2/sam/sam_vit_b_01ec64.pth"
        lora_ckpt = None
        rank = 4
        n_gpu = 1
        batch_size = 1
        enable_memory = True
        visualize = False
        prompt_type = 'bbox'
        semi = False
        reinforce = False
        multi_prompts = False
        use_checkpoint = False
    
    class Opt:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        img_size = 256
        pre_trained = False
        classes = 2
        
    args = Args()
    opt = Opt()
    
    model = get_model("MemSAM", args=args, opt=opt)
    model.to(opt.device)
    model.eval() # Important for inference test
    
    imgs = batch['image'].to(dtype=torch.float32, device=opt.device)
    bboxes = batch['bbox'].to(dtype=torch.float32, device=opt.device)
    
    print(f"Input image type: {type(imgs)}, shape: {imgs.shape}")
    print(f"Input bbox type: {type(bboxes)}, shape: {bboxes.shape}")
    
    with torch.no_grad():
        try:
            out = model(imgs, pt=None, bbox=bboxes)
            print(f"Model Forward Succeeded!")
            
            if isinstance(out, dict):
                print(f"Output keys: {out.keys()}")
                if 'masks' in out:
                    mask_tensor = out['masks']
                    # Use a safer way to get shape to avoid indexing warning
                    m_shape = tuple(mask_tensor.shape)
                    print(f"Masks shape: {m_shape}")
                    
                    if mask_tensor.shape[1] == 1:
                        print("SUCCESS: 1-channel binary output verified.")
            else:
                print(f"Output type is {type(out)}, not dict.")
        except Exception as e:
            print(f"Model Forward Failed with error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    verify_ovr_setup()
