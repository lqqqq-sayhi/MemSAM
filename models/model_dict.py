import torch.nn as nn
from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_memsam.build_memsam import memsam_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
        
        # Unfreeze mask_decoder parameters for fine-tuning
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
        print("[get_model] Unfrozen mask_decoder parameters.")
            
    elif modelname == "MemSAM":
        model = memsam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
        # Unfreeze mask_decoder parameters for fine-tuning
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
        print("[get_model] MemSAM: Unfrozen mask_decoder parameters.")
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
