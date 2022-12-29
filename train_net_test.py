#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances, load_coco_json

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2 import model_zoo

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
output_dir = "./output/IS_test_20221228"

num_classes = 100
device ="cuda" # "cpu"
cfg_save_path = "IS_cfg.pickle"
cur_dir = "/home/smartcoop/test/detectron2/52_coco_data"
train_dir = os.path.join(cur_dir, "train")
test_dir = os.path.join(cur_dir, "val")
val_test_dir = os.path.join(cur_dir, "test")


# Training Testing dataset
# change
"""
train_dataset_name = "agv_train_20221117"
train_images_path = "/media/hp/DATADRIVE1/xrProject_Datasets/agv_dataset/output/training/images/"
train_json_annotation_path = "/media/hp/DATADRIVE1/xrProject_Datasets/agv_dataset/output/training/coco_instances.json"

test_dataset_name = "agv_val_20221117"
test_images_path = "/media/hp/DATADRIVE1/xrProject_Datasets/agv_dataset/output/val/images/"
test_json_annotation_path = "/media/hp/DATADRIVE1/xrProject_Datasets/agv_dataset/output/val/coco_instances.json"
"""

#register_coco_instances(name = test_dataset, metadata={}, json_file=test_json, image_root=test_images)

train_dataset_name = "100_train"
test_dataset_name = "100_val"
test_dataset = "100_test"

train_json_annotation_path = os.path.join(train_dir, "datasets.json")
train_images_path = os.path.join(train_dir)

test_json_annotation_path = os.path.join(test_dir, "datasets.json")
test_images_path = os.path.join(test_dir)

test_json = os.path.join(val_test_dir, "datasets.json")
test_images = os.path.join(val_test_dir)

#register_coco_instances(training_dataset_name, {}, training_json_file, training_img_dir)
#training_dict = load_coco_json(training_json_file, training_img_dir,
#                dataset_name=training_dataset_name)
#training_metadata = MetadataCaalog.get(training_dataset_name)

#register_coco_instances(test_dataset_name, {}, test_json_file, test_img_dir)
#test_dict = load_coco_json(test_json_file, test_img_dir,
#                dataset_name=test_dataset_name)
#test_metadata = MetadataCatalog.get(test_dataset_name)


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg =get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    # cfg.DATALOADER.NUM_WORKERS = 2

    # cfg.SOLVER.IMS_PER_BATCH = 2

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300000
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 50000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # cfg = setup(args)
    register_coco_instances(train_dataset_name, {}, train_json_annotation_path, train_images_path)
    register_coco_instances(test_dataset_name, {}, test_json_annotation_path, test_images_path)
    register_coco_instances(test_dataset, {}, test_json, test_images)
    
    training_dict = load_coco_json(train_json_annotation_path, train_images_path, train_dataset_name)
    val_dict = load_coco_json(test_json_annotation_path, test_images_path, test_dataset_name)
    test_dict = load_coco_json(test_json, test_images, test_dataset)

    training_metadata = MetadataCatalog.get(train_dataset_name)
    val_metadata = MetadataCatalog.get(test_dataset_name)
    test_metadata = MetadataCatalog.get(test_dataset)

    cfg  = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True) #True
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )