import pandas as pd
from pathlib import Path
import sys
sys.path.append('/home/schnablelab/cmiao/MyRepo/schnablelab/apps')
from Tools import GenDataFrameFromPath

data_dir = '/lustre/work/schnablelab/cmiao/class_879/cse_479/project/data/dataset_builder_g6'
data_dir_path = Path(data_dir)
df = GenDataFrameFromPath(data_dir_path/'train')
print('train: %s'%df.shape[0])
df['label'] = df['fn'].apply(lambda x: x.split('_')[2][1])
df[['fn', 'label']].to_csv('train_label.csv', header=None, index=False)
df = GenDataFrameFromPath(data_dir_path/'test')
print('test: %s'%df.shape[0])
df['label'] = df['fn'].apply(lambda x: x.split('_')[2][1])
df[['fn', 'label']].to_csv('test_label.csv', header=None, index=False)
