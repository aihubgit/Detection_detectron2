import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#import matplolib.pyplot as plt
import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances, load_coco_json

from detectron2.engine import DefaultTrainer
import os
import time
import tqdm


config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#output_dir = "./output/IS_test_20221219"


"""
cur_dir = "/home/smartcoop/test/detectron2/52_coco_data/"
train_dir = os.path.join(cur_dir, "train")
test_dir = os.path.join(cur_dir, "val")
dir_t = os.path.join(cur_dir, "test")

training_dataset_name = "100_train"
test_dataset_name = "100_val"
test_name = "100_test"

training_json_file = os.path.join(train_dir, "datasets.json")
training_img_dir = os.path.join(train_dir)

test_json_file = os.path.join(test_dir, "datasets.json")
test_img_dir = os.path.join(test_dir)

json_file = os.path.join(dir_t, "datasets.json")
img_dir = os.path.join(dir_t)

register_coco_instances(training_dataset_name, {}, training_json_file, training_img_dir)
training_dict = load_coco_json(training_json_file, training_img_dir, dataset_name=training_dataset_name)
training_metadata = MetadataCatalog.get(training_dataset_name)

register_coco_instances(test_dataset_name, {}, test_json_file, test_img_dir)
test_dict = load_coco_json(test_json_file, test_img_dir, dataset_name=test_dataset_name)
test_metadata = MetadataCatalog.get(test_dataset_name)
#.set(thing_classes = ["당구 초크", "레몬", "소프트럭비공", "주사위"])

register_coco_instances(test_name, {}, json_file, img_dir)
ts_dict = load_coco_json(json_file, img_dir, dataset_name=test_name)
ts_metadata = MetadataCatalog.get(test_name)
"""


cfg = get_cfg()
cfg.DATASETS.TRAIN = ("100_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 8
cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 300000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 100  # 1 classes (person)



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg)
#trainer.resume_or_load(resume=True)
# trainer.train()
cfg.MODEL.WEIGHTS = "/home/smartcoop/test/detectron2/output/IS_test_20221228/model_0079999.pth" # 여기부분은 본인의 model이저장된 경로로 수정해줍니다.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
cfg.DATASETS.TEST = ("100_val", )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import itertools
from detectron2.utils.file_io import PathManager
import json

path = "/home/smartcoop/test/detectron2/52_coco_data/sample/"
#수정 필요
files = os.listdir(path)
predictions = []
img = []
print(files)


def getCetIdMap():
    categoryPath = '/home/smartcoop/다운로드/datasets - $.categories.json' # 카테고리B 패스 (훈련때 사용한 categories)
    
    with open(categoryPath, 'r') as f:
         categoryData = json.load(f)
    
    ids = [category['id'] for category in categoryData]
    ids = sorted(ids)
    id_map = {i: v for i, v in enumerate(ids)}

    cats = {}
    for cat in categoryData:
        cats[cat['id']] = cat
        
    cats = [cats[id] for id in ids]
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    
    MetadataCatalog.get('inference').set(thing_classes=thing_classes)
        
    return id_map

idmap = getCetIdMap()
start = time.time()

with tqdm.tqdm(total=len(files), desc="inference") as p:
    for index, i in enumerate(files):
        images = {}
        # realIndex = index+9
        if '.jpg' in i:
            image = os.path.join(path, i)
            im = cv2.imread(image)
            outputs = predictor(im)

            predictions.extend(instances_to_coco_json(outputs["instances"].to('cpu'), index))
            images["id"] = index
            images["file_name"] = i
            img.append(images)
            #v = Visualizer(im[:,:,::-1], metadata=MetadataCatalog.get('inference'))
            #v = v.draw_instance_predictions(outputs["instances"].to('cpu'))
            #v.save("output/IS_test_20221228/inference_test2/{0}".format(i))
        p.update()

for v in predictions:
   v['category_id'] = idmap[v['category_id']]

test_output_dir = "/home/smartcoop/test/detectron2/output/IS_test_20221228/test_output2/result.json"
with PathManager.open(test_output_dir, 'w') as f:
    json.dump(predictions, f, ensure_ascii=False)

test_image_output_dir = "/home/smartcoop/test/detectron2/output/IS_test_20221228/test_output2/images.json"
with PathManager.open(test_image_output_dir, 'w') as f:
    json.dump(img, f, ensure_ascii=False)


"""
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


evaluator = COCOEvaluator("100_test", ("bbox","segm"), False, output_dir= "/home/smartcoop/test/detectron2/output/IS_test_20221228/test_output/")
val_loader = build_detection_test_loader(cfg, "100_train")

inference_on_dataset(predictor.model, val_loader, evaluator)
"""
end = time.time()

print(f'{end-start} sec')
