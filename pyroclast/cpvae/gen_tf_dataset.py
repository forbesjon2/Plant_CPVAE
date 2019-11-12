import tensorflow_datasets as tfds
from pathlib import Path
import tensorflow as tf
from matplotlib.pyplot import imread
import numpy as np
import os
#import matplotlib.pyplot as plt
#from pyroclast.common.util import img_preprocess
#tf.compat.v1.enable_eager_execution()

class DatasetBuilderG6(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.1.0")
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Thsi is the dataset for group 6 CPVAE project"),
            features=tfds.features.FeaturesDict({
                "image_description": tfds.features.Text(),
                "image": tfds.features.Image(shape=(128,128,3), dtype=tf.uint8),
                "label": tfds.features.ClassLabel(num_classes=2),
            }),
            supervised_keys=("image", "label"),
        )
    def _split_generators(self, dl_manager):
        data_dir = dl_manager.manual_dir
        
        return[
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs={
                    "images_dir_path": os.path.join(data_dir, 'train'),
                    "labels": os.path.join(data_dir, 'train_label.csv'),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=4,
                gen_kwargs={
                    "images_dir_path": os.path.join(data_dir,'test'),
                    "labels": os.path.join(data_dir, 'test_label.csv'),
                },
            ),
        ]
    def _generate_examples(self, images_dir_path, labels):
        with tf.io.gfile.GFile(labels) as f:
            for i in f:
                img_fn, label_str = i.rstrip().split(',')
                img_fn_path = os.path.join(images_dir_path, img_fn)
                label = 0 if label_str=='1' else 1
                record = {
                    'image_description': img_fn,
                    'image': img_fn_path,
                    'label': label,
                }
                yield img_fn, record

def gen_data_dict(manual_dir='/work/schnablelab/cmiao/class_879/Project/Data', 
		download_dir='/work/schnablelab/cmiao/class_879/Project/Data/downloaded_dir'):
    dataset = DatasetBuilderG6()
    dc = tfds.download.DownloadConfig(manual_dir=manual_dir)
    dataset.download_and_prepare(download_config=dc, download_dir=download_dir)
    g6 = dataset.as_dataset(batch_size=100, shuffle_files=True)
    #g6['train'] = g6['train'].map(lambda x: img_preprocess(x, 128)).batch(50)
    #g6['test'] = g6['test'].map(lambda x: img_preprocess(x, 128)).batch(50)
    print(dataset.info)
    return g6
