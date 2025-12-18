import json
import os
import random
import pickle
import gzip
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F
import json

from PIL import Image
from typing import Optional
from pathlib import Path
from datasets.tardataset import TarDataset
from glob import glob

from datasets.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from misc.depth import estimate_depth_scale_ransac
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir

from .colmap_utils import *
from .colmap_misc import *


def load_seq_data(data_path, split):
    file_path = data_path / f"{split}.pickle.gz"
    with gzip.open(file_path, "rb") as f:
        seqData:dict = pickle.load(f)
    # add a seq that check if the video is exist
    delSeqs=list()

    def isDirEmpty(folder_path)->bool:
        folder = Path(folder_path)
        return not any(folder.iterdir())

    for seqKey in seqData.keys():
        img_path = data_path/split/seqKey
        if(not os.path.isdir(img_path) or isDirEmpty(img_path)):
               delSeqs.append(seqKey)
    print(f"[WARNING] {len(delSeqs)} seq(s) loss!")
    for delSeq in delSeqs:
        seqData.pop(delSeq)
    return seqData


class LLFF(data.Dataset):
    def __init__(self,
                 cfg,
                 split: Optional[str]=None,
                ) -> None:
        self.cfg=cfg
        self.is_train=False
        self.dataPath=Path("/media/yzh/Dataset3/3DReconstruction/4_llff_output_mirror/llff")
        self.posePath=Path("/media/yzh/Dataset3/3DReconstruction/4_nerf_llff_data")
        self.poseData=list() # 通过json加载的位姿信息
        self.image_size=[504,378]

        self.num_scales = len([0])
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)
        return
            
    def __len__(self) -> int:
        return self.length
    
    def _load_seq_data(self, split):
        return load_seq_data(self.data_path, split)
 
    def _full_index(self, seq_keys, seq_data, left_offset, extra_frames):
        skip_bad = self.cfg.dataset.skip_bad_shape
        if skip_bad:
            fn = self.data_path / "valid_seq_ids.train.pickle.gz"
            valid_seq_ids = pickle.load(gzip.open(fn, "rb"))
        key_id_pairs = []
        for seq_key in seq_keys:
            seq_len = len(seq_data[seq_key]["timestamps"])
            frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
            if skip_bad:
                good_frames = valid_seq_ids[seq_key]
                frame_ids = [f_id for f_id in frame_ids if f_id in good_frames]
            seq_key_id_pairs = [(seq_key, f_id) for f_id in frame_ids]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs
    
    def _load_sparse_pcl(self, seq_key):
        fn = f"pcl.{self.split_name_for_loading}/{seq_key}.pickle.gz"
        f = self.pcl_dataset.get_file(fn)
        data = gzip.decompress(f.read())
        return pickle.loads(data)

        
    def _load_image(self, key, id):
        suffix=""
        for suffix_t in ["jpg","JPG","png","PNG"]:
            if(self.data_path/key/"images"/f"{id}.{suffix}").exists():
                suffix=suffix_t
                break
        assert suffix != ""
        img = self.loader(self.data_path/key/"images"/f"{id}.{suffix}")  # TODO：验证大小是否合适
        img.resize(size=self.image_size)
        return img
    
    def _load_depth(self, key, id):

        suffix=""
        for suffix_t in ["jpg","JPG","png","PNG"]:
            if(self.data_path/key/"depth_maps"/f"{id}.{suffix}").exists():
                suffix=suffix_t
                break
        assert suffix != ""
        depth = Image.open(self.data_path/key/"depth_maps"/f"{id}.{suffix}")
        # Scale the saved image using the metadata
        max_value = float(depth.info["max_value"])
        min_value = float(depth.info["min_value"])
        # Scale from uint16 range
        depth = (np.array(depth).astype(np.float32) / (2 ** 16 - 1)) * (max_value - min_value) + min_value
        depth.resize(self.image_size)
        return depth
    
    @staticmethod
    def _load_split_indices(index_path):
        "load the testing split from txt"
        def get_key_id(s):
            parts = s.split(" ")
            key = parts[0]
            src_idx = int(parts[1])
            tgt_5_idx = int(parts[2])
            tgt_10_idx = int(parts[3])
            tgt_random_idx = int(parts[4])
                                                      
            return key, [src_idx, tgt_5_idx, tgt_10_idx, tgt_random_idx]

        with open(index_path, "r") as f:
            lines = f.readlines()
        key_id_pairs = list(map(get_key_id, lines))
        return key_id_pairs

    def get_frame_data(self, seq_key, frame_idx, color_aug_fn):
        # load the pose
        with open(str(self.posePath/seq_key/"cameras.json")) as f:
            poseDataRaws=json.load(f)
        assert len(poseDataRaws)>=3
        i=1
        self.poseData.clear()
        while(poseDataRaws[i]["img_name"]>poseDataRaws[i-1]["img_name"]):
            i+=1
        while (poseDataRaws[i]["img_name"] > poseDataRaws[i - 1]["img_name"]):
            self.poseData.append(poseDataRaws[i])
            i += 1
        index=frame_idx % (len(self.poseData))
        frame_name=self.poseData[index]
        # load the image
        img = self._load_image(seq_key, frame_name)  # PILLOW format
        # load pre-process depth for training
        if self.dataPath is not None:
            depth = self._load_depth(seq_key, frame_idx)
            if depth is not None:
                depth = self.to_tensor(depth)
                depth = F.interpolate(depth[None,...], size=self.image_size, mode="nearest")[0]
        else:
            depth = None

        # load the intrinsics matrix
        cameras=read_cameras_binary(self.data_path/seq_key/"sparse"/"0"/"cameras.bin")
        K = read_camera_params(cameras[0])
        # load the extrinsic matrixself.num_scales
        w2c=np.zeros([4,4],dtype=np.float64)
        pos=np.asarray(self.poseData[index]["position"],dtype=np.float64)
        rot=np.asarray(self.poseData[index]["rotation"],dtype=np.float64)
        w2c[0:3,0:3]=pos
        w2c[3,0:3]=rot.transpose()
        c2w = data_to_c2w(w2c)
        img_scale = self.resize[0](img)
        inputs_color = self.to_tensor(img_scale)
        if self.cfg.dataset.pad_border_aug != 0:
            inputs_color_aug = self.to_tensor(color_aug_fn(self.pad_border_fn(img_scale)))
            if depth is not None:
                pad = self.cfg.dataset.pad_border_aug
                depth = F.pad(depth, (pad,pad,pad,pad), mode="replicate")
        else:
            inputs_color_aug = self.to_tensor(color_aug_fn(img_scale))

        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1]
        K_scale_target[1, :] *= self.image_size[0]
        # scale K_inv for unprojection according to how much padding was added
        K_scale_source = K.copy()
        # scale focal length by size of original image, scale principal point for the padded image
        K_scale_source[0, 0] *=  self.image_size[1]
        K_scale_source[1, 1] *=  self.image_size[0]
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg.dataset.pad_border_aug * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg.dataset.pad_border_aug * 2)
        inv_K_source = np.linalg.pinv(K_scale_source)

        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        # original world-to-camera matrix in row-major order and transfer to column-major order
        inputs_T_c2w = torch.from_numpy(c2w)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source, inputs_color, inputs_color_aug, inputs_T_c2w, img.size, depth



    def getItemID(self, id):
        inputs = {}

        inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_color, inputs_color_aug, \
        inputs_T_c2w, orig_size, inputs_depth = self.get_frame_data(seq_key=id,
                                                                    frame_idx=0,
                                                                    color_aug_fn=False
                                                                    )

        if self.cfg.dataset.scale_pose_by_depth:
            # get colmap_image_id
            xyd = get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx)
        else:
            xyd = None

        # input_frame_idx = src_and_tgt_frame_idxs[0]
        timestamp = self._seq_data[seq_key]["timestamps"][frame_idx]
        inputs[("frame_id", 0)] = f"{self.split_name_for_loading}+{seq_key}+{timestamp}"

        inputs[("K_tgt", frame_name)] = inputs_K_tgt
        inputs[("K_src", frame_name)] = inputs_K_src
        inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
        inputs[("color", frame_name, 0)] = inputs_color
        inputs[("color_aug", frame_name, 0)] = inputs_color_aug
        # original world-to-camera matrix in row-major order and transfer to column-major order
        inputs[("T_c2w", frame_name)] = inputs_T_c2w
        inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)
        if inputs_depth is not None:
            inputs[("unidepth", frame_name, 0)] = inputs_depth

        if xyd is not None and frame_name == 0:
            inputs[("depth_sparse", frame_name)] = xyd

            if inputs_depth is not None and self.cfg.dataset.ransac_on_the_fly:
                _, H, W = inputs_depth.shape
                inputs[("scale_colmap", frame_name)] = estimate_depth_scale_ransac(
                    inputs_depth.unsqueeze(0)[:,
                    self.cfg.dataset.pad_border_aug:H - self.cfg.dataset.pad_border_aug,
                    self.cfg.dataset.pad_border_aug:W - self.cfg.dataset.pad_border_aug],
                    inputs[("depth_sparse", frame_name)]
                )

        # ==== Load llava_feats ====
        max_token_len = 39
        try:
            llava_feat = self.llava_feats[seq_key]
        except:
            llava_feat = None

        if llava_feat is not None:
            l_feat = np.load(llava_feat)
            l_feat = torch.from_numpy(l_feat)

            if l_feat.shape[0] < max_token_len:
                rest = max_token_len - l_feat.shape[0]
                pad = torch.zeros([rest, l_feat.shape[1]])
                l_feat = torch.cat([l_feat, pad], dim=0)
        else:
            l_feat = torch.zeros([39, 5120])

        inputs[('llava_feat', 0)] = l_feat

        return inputs

    def __getitem__(self, index):
        inputs = {}

        # random frame sampling
        if self.is_train:
            # train data contains pairs of sequence name, source frame index
            seq_key, src_idx = self._seq_key_src_idx_pairs[index]
            pose_data = self._seq_data[seq_key]
            seq_len = len(pose_data["timestamps"])

            if self.cfg.dataset.frame_sampling_method == "two_forward_one_back":
                if self.dilation == "random":
                    dilation = torch.randint(1, self.max_dilation, (1,)).item()
                    left_offset = dilation # one frame in the past
                else:
                    # self.dilation and self._left_offsets can be fixed if cfg.dataset.dilation is an int
                    dilation = self.dilation
                    left_offset = self._left_offset
                # frame count is num_novel_frames + 1 for source view
                # sample one frame in backwards time and self.frame_count - 2 into the future
                src_and_tgt_frame_idxs = [src_idx - left_offset + i * dilation for i in range(self.frame_count)]
                # reorder and make sure indices don't go beyond start or end of the sequence
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i, seq_len-1), 0) for i in src_and_tgt_frame_idxs if i != src_idx]
            elif self.cfg.dataset.frame_sampling_method == "random":
                # random indices between -30 and 30 which will mean the offset 
                target_frame_idxs = torch.randperm( 4 * self.max_dilation + 1 )[:self.frame_count] - 2 * self.max_dilation
                # check that 0 is not included and that the indides dont go beyond the end of the sequence
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i + src_idx, seq_len-1), 0) for i in target_frame_idxs.tolist() if i != 0][:self.frame_count - 1]                
            frame_names = [0] + self.novel_frames

        # load src, 5 frames into future, 10 frames into future and random
        # follows MINE split and evaluation protocol
        else:
            # test data contains pairs of sequence name, [src_idx, tgt_idx1, tgt_idx2, tgt_idx3]
            seq_key, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]
            pose_data = self._seq_data[seq_key]

            frame_names = [0, 1, 2, 3]

        if self.cfg.dataset.scale_pose_by_depth:
            sparse_pcl = self._load_sparse_pcl(seq_key)

        # load the data
        do_color_aug = self.is_train and random.random() > 0.5 and self.color_aug
        if do_color_aug:
            color_aug = T.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):
            try:
                inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_color, inputs_color_aug, \
                inputs_T_c2w, orig_size, inputs_depth = self.get_frame_data(seq_key=seq_key, 
                                                    frame_idx=frame_idx, 
                                                    pose_data=pose_data,
                                                    color_aug_fn=color_aug
                )
            except:
                print(seq_key, "doesn't exist...!!!!")
                continue
            
            if self.cfg.dataset.scale_pose_by_depth:
                # get colmap_image_id
                xyd = get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx)
            else:
                xyd = None
            
            # input_frame_idx = src_and_tgt_frame_idxs[0]
            timestamp = self._seq_data[seq_key]["timestamps"][frame_idx]
            inputs[("frame_id", 0)] = f"{self.split_name_for_loading}+{seq_key}+{timestamp}"
            
            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            # original world-to-camera matrix in row-major order and transfer to column-major order
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)
            if inputs_depth is not None:
                inputs[("unidepth", frame_name, 0)] = inputs_depth

            if xyd is not None and frame_name == 0:
                inputs[("depth_sparse", frame_name)] = xyd

                if inputs_depth is not None and self.cfg.dataset.ransac_on_the_fly:
                    _, H, W = inputs_depth.shape
                    inputs[("scale_colmap", frame_name)] = estimate_depth_scale_ransac(
                        inputs_depth.unsqueeze(0)[:, 
                            self.cfg.dataset.pad_border_aug:H-self.cfg.dataset.pad_border_aug,
                            self.cfg.dataset.pad_border_aug:W-self.cfg.dataset.pad_border_aug],
                        inputs[("depth_sparse", frame_name)]
                    )

            # ==== Load llava_feats ====
            max_token_len = 39
            try:
                llava_feat = self.llava_feats[seq_key]
            except:
                llava_feat = None
                
            if llava_feat is not None:
                l_feat = np.load(llava_feat)
                l_feat = torch.from_numpy(l_feat)
                
                if l_feat.shape[0] < max_token_len:
                    rest = max_token_len - l_feat.shape[0]
                    pad = torch.zeros([rest, l_feat.shape[1]])
                    l_feat = torch.cat([l_feat, pad], dim=0)
            else:
                l_feat = torch.zeros([39, 5120])
                
            inputs[('llava_feat', 0)] = l_feat

        return inputs
