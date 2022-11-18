PACKAGE = D:\GitHub\Classification-and-Localization

default: run

run: workspace\training_demo\python TF-image-od.py
	workspace\training_demo\python TF-image-od.py

train: workspace\training_demo\model_main_tf2.py
	python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config

expo: workspace\training_demo\exporter_main_v2.py
	python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model

build: ${PACKAGE}\workspace\training_demo\images\train ${PACKAGE}\workspace\training_demo\annotations\label_map.pbtxt
	python generate_tfrecord.py -x D:\GitHub\Classification-and-Localization\workspace\training_demo\images\train -l D:\GitHub\Classification-and-Localization\workspace\training_demo\annotations\label_map.pbtxt -o D:\GitHub\Classification-and-Localization\workspace\training_demo\annotations\train.record
	python generate_tfrecord.py -x D:\GitHub\Classification-and-Localization\workspace\training_demo\annotations\label_map.pbtxt -o D:\GitHub\Classification-and-Localization\training_demo\annotations\test.record	