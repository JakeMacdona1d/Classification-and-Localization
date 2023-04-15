import os
from tensorflow.python.keras.utils.data_utils import get_file

# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz"
fileName = os.path.basename(modelURL)
cacheDir = "./pretrainedModels"
os.makedirs(cacheDir, exist_ok=True)

get_file(fname=fileName,
origin=modelURL, cache_dir=cacheDir, cache_subdir="checkpoints",extract= True)


# research 
# efficient det is best according to :
# https://medium.com/codex/review-efficientdet-scalable-and-efficient-object-detection-ed9ebc70f873
# https://victordibia.com/blog/state-of-object-detection/#two-stage-detection 