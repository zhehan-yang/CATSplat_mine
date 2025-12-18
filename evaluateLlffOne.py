import os
import json
import hydra
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

from models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from misc.util import add_source_frame_id
from misc.visualise_3d import save_ply

from datasets.llff import *

def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def evaluateOnlyOne(model, cfg, evaluator, inputs, device=None, save_vis=False):
    model_model = get_model_instance(model)
    model_model.set_eval()

    score_dict = {}

    if save_vis:
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        print(f"saving images to: {out_dir.resolve()}")
        seq_name = inputs["seq_name"]
        out_out_dir = out_dir / seq_name
        out_out_dir.mkdir(exist_ok=True)
        out_pred_dir = out_out_dir / f"pred"
        out_pred_dir.mkdir(exist_ok=True)
        out_gt_dir = out_out_dir / f"gt"
        out_gt_dir.mkdir(exist_ok=True)
        out_dir_ply = out_out_dir / "ply"
        out_dir_ply.mkdir(exist_ok=True)

    with torch.no_grad():
        if device is not None:
            to_device(inputs, device)
        target_frame_ids=[1,2,3]
        inputs["target_frame_ids"] = target_frame_ids
        outputs = model(inputs)

    for f_id in score_dict.keys():
        pred = outputs[('color_gauss', f_id, 0)]
        if cfg.dataset.name == "dtu":
            gt = inputs[('color_orig_res', f_id, 0)]
            pred = TF.resize(pred, gt.shape[-2:])
        else:
            gt = inputs[('color', f_id, 0)]
        # should work in for B>1, however be careful of reduction
        out = evaluator(pred, gt)
        if save_vis:
            save_ply(outputs, out_dir_ply / f"{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
            pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
            plt.imsave(str(out_gt_dir / f"{f_id:03}.png"), gt)

@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    loaded_optim, loaded_step, loaded_scheduler = model.load_model(cfg.run.checkpoint, ckpt_ids=0)
    
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    split = "test"
    save_vis = cfg.eval.save_vis
    inputs=dict()

    datasetloader=LLFF()
    inputs=datasetloader.getItemID("fern")


    evaluateOnlyOne(model, cfg, evaluator,inputs,device=device, save_vis=save_vis)
    

if __name__ == "__main__":
    main()
