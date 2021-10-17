# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from d2.detr import DetrDatasetMapper, add_detr_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from datetime import datetime
import pdb

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if "Detr" == cfg.MODEL.META_ARCHITECTURE:
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    now = datetime.now()
    curtime = now.isoformat()
    curtime = curtime.replace('-', '')
    curtime = curtime.replace(':', '')[2:13]

    if args.eval_only:
        logroot = './log/_eval'
        dirname = args.modelweight.split('/')[-2]

        if args.imagesave:
            args.log_dir = '{logroot}/imgs_eval_{dirname}_{thr:.02f}_{gpu}'.format(
                logroot=logroot, dirname=dirname, gpu=args.gpu, thr=args.testthr
            )
        else:
                
            args.log_dir = '{logroot}/eval_{dirname}_{thr:.02f}_{gpu}'.format(
                logroot=logroot, dirname=dirname, gpu=args.gpu, thr=args.testthr
            )
    else:
        logroot = './log/'

        if args.resume:
            dirname = "/".join(args.modelweight.split('/')[:-1])
            args.log_dir = '{dirname}_{gpu}'.format(
                dirname=dirname, gpu=args.gpu
            )
        else:
            if args.colormode != '':
                colormode = '_' + args.colormode
            else:
                colormode = args.colormode
            setname = '' if args.setname == '' else '_' + args.setname
            args.log_dir = '{logroot}{dataname}{setname}{colormode}_{lr:.07f}_{postfix}{time}_{gpu}'.format(
                logroot=logroot, dataname=args.dataname, setname=setname, time=curtime,
                postfix=args.postfix, gpu=args.gpu, colormode=colormode, lr=cfg.SOLVER.BASE_LR,
            )

    add_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    root = '{0}/dataset/endoscopy/data_internal/detection/'.format(
        os.getcwd()[:os.getcwd().find('/python/')]
    )
    print(root)
    setname = '' if args.setname == '' else '_' + args.setname
    colormode = '' if args.colormode == '' else '_' + args.colormode

    cfg.DATASETS.TRAIN = ("{0}_train".format(args.dataname),)
    cfg.DATASETS.TEST = ("{0}_test".format(args.dataname),)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = args.batchsize
    # '''
    # num_gpu = 1
    # bs = (num_gpu * 2)
    # cfg.SOLVER.BASE_LR = 0.01 * bs / 16  # pick a good LR
    # '''
    # # cfg.SOLVER.BASE_LR = 2.5e-6  # pick a good LR
    
    # cfg.SOLVER.BASE_LR = args.lr  # 2.5e-4 pick a good LR

    if args.maxiter > 0:
        cfg.SOLVER.MAX_ITER = args.maxiter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
###########

    if args.colormode == '':
        colormode = 'YOLO'
    else:
        colormode = '{0}_CHE'.format(args.colormode)

    register_coco_instances("{0}_train".format(args.dataname), {},
                            root + "{dataname}/{dataname}{setname}_train.json".format(
                                dataname=args.dataname, setname=setname),
                            root + "{dataname}/{dataname}_{colormode}/train".format(
                                dataname=args.dataname, colormode=colormode))
    register_coco_instances("{0}_test".format(args.dataname), {},
                            root + "{dataname}/{dataname}{setname}_test.json".format(
                                dataname=args.dataname, setname=setname),
                            root + "{dataname}/{dataname}_{colormode}/test".format(
                                dataname=args.dataname, colormode=colormode))
                                
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.numclasses  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    
    # setup_logger(args.log_dir)
    cfg.OUTPUT_DIR = args.log_dir
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def main_worker(args):
    cfg = setup(args)
    # 저 setup 안에 output_dir 잇고 setup_logger 도 있다.

    # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
#########
    # if args.modelweight.find('modelzoo') > -1:
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.modelname)
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    # elif args.modelweight == '':
    #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.modelname)  # Let training initialize from model zoo
    # else:
    #     cfg.MODEL.WEIGHTS = args.modelweight


    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    print(cfg)
    print('=====================')
    print(cfg.SOLVER.MAX_ITER)
    print(cfg.SOLVER.STEPS)
    print(cfg.OUTPUT_DIR)
    
    return trainer.train()

def main():
    parser = default_argument_parser()
    parser.add_argument(
        '--dataname', type=str,
        default='labeling_211007')
    parser.add_argument(
        '--batchsize', type=int,
        default='4')
    parser.add_argument(
        '--gpu', type=int,
        default='4')
    parser.add_argument(
        '--numclasses', type=int,
        default='4')
    parser.add_argument(
        '--maxiter', type=int,
        default='0')
    parser.add_argument(
        '--setname', type=str,
        default='')
    parser.add_argument(
        '--colormode', type=str,
        default='')
    parser.add_argument(
        '--postfix', type=str,
        default='')
    parser.add_argument(
        '--log-dir', type=str,
        default='')
    parser.add_argument(
        '--modelweight', type=str,
        default='')
    parser.add_argument('--imagesave', action='store_true')
    # '''
    # num_gpu = 1
    # bs = (num_gpu * 2)
    # cfg.SOLVER.BASE_LR = 0.01 * bs / 16  # pick a good LR
    # '''
    # # cfg.SOLVER.BASE_LR = 2.5e-6  # pick a good LR
###########
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(args.gpu)

    args.num_gpus = args.gpu.count(",") + 1
    args.num_machines = args.gpu.count(",") + 1
    print("Command Line Args:", args)
    launch(
        main_worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

if __name__ == "__main__":
    main()
