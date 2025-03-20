# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import random_split, DistributedSampler
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
from AsymKD.util.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import Compose
import cv2
import json
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
import torch.nn as nn


class StereoDataset(data.Dataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.crop_size = None
        if aug_params is not None and "crop_size" in aug_params:
            #self.augmentor = SparseFlowAugmentor(**aug_params)
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
            self.crop_size = [aug_params['crop_size'][0],aug_params['crop_size'][1]]
            self.resize = Resize(
                    height=aug_params['crop_size'][0],
                    width=aug_params['crop_size'][1],
                    image_interpolation_method=cv2.INTER_CUBIC,
                )
            self.transform = Compose([
                self.resize,
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        else:
            self.transform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        
        self.segment_anything_predictor = seg_any_predictor
        

    def check_item(self):
        if(len(self.disparity_list)!=len(self.image_list)):
            print("Error : ", len(self.disparity_list),len(self.image_list))

    def __getitem__(self, index):
        try:
            if self.is_test:
                img1 = cv2.imread(self.image_list[index])
                return img1, self.extra_info[index]

            if not self.init_seed:
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    torch.manual_seed(worker_info.id)
                    np.random.seed(worker_info.id)
                    random.seed(worker_info.id)
                    self.init_seed = True

            index = index % len(self.image_list)
            disp = self.disparity_reader(self.disparity_list[index])
            img1 = cv2.imread(self.image_list[index])

            if self.augmentor is not None:
                height, width = img1.shape[:2]
                if height < width:
                    new_height = 518
                    new_width = int((new_height / height) * width)
                else:
                    new_width = 518
                    new_height = int((new_width / width) * height)

                # 이미지 크기 조정
                img1 = cv2.resize(img1, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                # 이미지 크기 조정
                resized_disp = cv2.resize(np.expand_dims(disp[0], axis=2), (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                resized_valid = cv2.resize(np.expand_dims(disp[1], axis=2).astype(np.float32), (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                resized_valid = resized_valid>0
                disp = (resized_disp,resized_valid)

            valid = None

            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 512

            flow = np.stack([disp, np.zeros_like(disp)], axis=-1)



            if self.augmentor is not None:
                depth_image, seg_image = None,None
            else:
                depth_image, seg_image = img1,img1

            # if self.augmentor is not None:
            #     depth_image, seg_image, flow, valid = self.augmentor(img1, img1, flow, valid)

            if self.augmentor is not None:
                if self.sparse:
                    depth_image, seg_image, flow, valid = self.augmentor(img1, img1, flow, valid)
                else:
                    depth_image, seg_image, flow = self.augmentor(img1, img1, flow)

            disp = np.array(disp).astype(np.float32)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB) / 255.0
            depth_image = self.transform({'image': depth_image})['image']
            depth_image = torch.from_numpy(depth_image)
            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
            seg_image = self.segment_anything_predictor.set_image(seg_image)
            seg_image = seg_image.squeeze(0)
            

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid)
            
            # if self.sparse:
            #     valid = torch.from_numpy(valid)
            # else:
            #     valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
            
            flow = flow[:1]
            
            
            return depth_image, seg_image, flow, valid.float()
        except Exception as e:
            filename = 'Exception_catch.txt'
            a = open(filename, 'a')

            # 새파일에 이어서 쓰기
            a.write(str(e)+' ' + self.image_list[index] + self.disparity_list[index] +'\n')
            a.close()
            return None

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset): #/home/wjchoi/data/BriGeS
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD', keywords=[]):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)
        
        root = osp.join(root, 'TartanAir')
        left_images = sorted( glob(osp.join(root, '*/*/*/image_left/*.png')) )
        right_images = sorted( glob(osp.join(root, '*/*/*/image_right/*.png')) )
        left_disparity_images = sorted( glob(osp.join(root, '*/*/*/depth_left/*.npy')) )
        right_disparity_images = sorted( glob(osp.join(root, '*/*/*/depth_right/*.npy')) )

        for img1, img2, disp1, disp2 in zip(left_images,right_images, left_disparity_images,right_disparity_images):
            self.image_list += [ img1 ]
            self.disparity_list += [ disp1 ]
            self.image_list += [ img2 ]
            self.disparity_list += [ disp2 ]
            if (img1.replace('image_left','depth_left').replace('left.png','left_depth.npy') != disp1):
                print("Error : ", img1, disp1)
                quit()
            if (img2.replace('image_right','depth_right').replace('right.png','right_depth.npy') != disp2):
                print("Error : ", img2, disp2)
                quit()

class KITTI(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD/kitti/kitti2015', image_set='training'):
        super(KITTI, self).__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp1_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png')))
        disp2_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_1/*_10.png')))

        for idx, (img1, img2, disp1,disp2) in enumerate(zip(image1_list, image2_list, disp1_list, disp2_list)):
            self.image_list += [ img1 ]
            self.disparity_list += [ disp1 ]
            self.image_list += [ img2 ]
            self.disparity_list += [ disp2 ]
            if (img1.replace('image_2','disp_occ_0') != disp1):
                print("Error : ", img1, disp1)
                quit()
            if (img2.replace('image_3','disp_occ_1') != disp2):
                print("Error : ", img2, disp2)
                quit()

class KITTI2012(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD/kitti/kitti2012', image_set='training'):
        super(KITTI2012, self).__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispKITTI2012)
        assert os.path.exists(root)

        image_list = sorted(glob(os.path.join(root, image_set, 'colored_0/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ/*_10.png')))

        for idx, (img, disp) in enumerate(zip(image_list, disp_list)):
            self.image_list += [ img ]
            self.disparity_list += [ disp ]
            if (img.replace('colored_0','disp_occ') != disp):
                print("Error : ", img, disp)
                quit()

class MegaDepth(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD/MegaDepth'):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispMegaDepth)
        assert os.path.exists(root)

        images = sorted( glob(osp.join(root, '*/*/imgs/*.jpg')) )
        disparities = sorted( glob(osp.join(root, '*/*/depths/*.h5')) )

        for img, disp in zip(images, disparities):
            self.image_list += [ img ]
            self.disparity_list += [ disp ]
            if (img.replace('imgs','depths').replace('jpg','h5') != disp):
                print("Error : ", img, disp)
                quit()

class HRWSI(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD/HRWSI'):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispHRWSI)
        assert os.path.exists(root)

        images = sorted( glob(osp.join(root, 'train','imgs/*.jpg')) )
        diparities = sorted( glob(osp.join(root, 'train', 'gts/*.png')) )

        for img, disp in zip(images, diparities):
            self.image_list += [ img ]
            self.disparity_list += [ disp ]
            if (img.replace('imgs','gts').replace('jpg','png') != disp):
                print("Error : ", img, disp)
                quit()

class BlendedMVS(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='/home/wodon326/datasets/AsymKD/BlendedMVS'):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispBlendedMVS)
        assert os.path.exists(root)

        # images = sorted( glob(osp.join(root, '*/blended_images/*_masked.jpg')) )
        # images = [img.replace('_masked', '') for img in images]
        images = sorted(glob(osp.join(root, '*/blended_images/*.jpg')))
        images = [img for img in images if '_masked' not in osp.basename(img)]
        images = sorted(images)
        disparities = sorted( glob(osp.join(root, '*/rendered_depth_maps/*.pfm')) )

        for img, disp in zip(images, disparities):
            self.image_list += [ img ]
            self.disparity_list += [ disp ]
            if (img.replace('blended_images','rendered_depth_maps').replace('jpg','pfm') != disp):
                print("Error : ", img, disp)
                quit()



class VKITTI(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='../../datasets/vkitti'):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispVKITTI)
        assert os.path.exists(root)

        with open('data_json/train_vkitti.json', 'r') as f:
            json_data = json.load(f)
        
        for file in json_data['files']:
            self.image_list += [root + '/' + file["rgb"] ]
            self.disparity_list += [ root+'/'+file["depth"] ]

class Hypersim(StereoDataset):
    def __init__(self, seg_any_predictor:SamPredictor, aug_params=None, root='../../datasets/Hypersim'):
        super().__init__(seg_any_predictor, aug_params, sparse=True, reader=frame_utils.readDispHypersim)
        assert os.path.exists(root)

        with open('data_json/train_hypersim.json', 'r') as f:
            json_data = json.load(f)
        
        for file in json_data['files']:
            self.image_list += [ root+'/'+file["rgb"] ]
            self.disparity_list += [ root+'/'+ file["depth"] ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
    # 데이터와 레이블을 분리하여 처리
    depth_image, seg_image, flow, valid = zip(*batch)
    depth_image = torch.stack(depth_image) 
    seg_image = torch.stack(seg_image) 
    flow = torch.stack(flow) 
    valid = torch.stack(valid) 
    return depth_image,seg_image,flow,valid
  
def fetch_dataloader(args, seg_any_predictor:SamPredictor, rank, world_size):
    """ Create the data loader for the corresponding trainign set """
    aug_params = None
    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
    
    #torch.multiprocessing.set_start_method('spawn')

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''))
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'VKITTI':
            new_dataset = VKITTI(seg_any_predictor, aug_params)
        elif dataset_name == 'Hypersim':
            new_dataset = Hypersim(seg_any_predictor, aug_params)
        elif dataset_name == 'HRWSI':
            new_dataset = HRWSI(seg_any_predictor, aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from HRWSI")
        elif dataset_name == 'MegaDepth':
            new_dataset = MegaDepth(seg_any_predictor, aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from MegaDepth")
        elif dataset_name == 'BlendedMVS':
            new_dataset = BlendedMVS(seg_any_predictor, aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from BlendedMVS")
        elif dataset_name == 'kitti':
            new_dataset = KITTI2012(seg_any_predictor, aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(seg_any_predictor, aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
        new_dataset.check_item()

    dataset_size = len(train_dataset)
    train_size = int(0.999 * dataset_size)
    val_size = dataset_size - train_size

    # 데이터셋 분할
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, num_workers = 4, sampler = sampler, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, drop_last=False, collate_fn=collate_fn, num_workers = 4)
    logging.info('Training with %d images' % len(train_dataset))
    return train_loader, val_loader
