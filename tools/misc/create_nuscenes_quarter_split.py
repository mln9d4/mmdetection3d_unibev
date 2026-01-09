import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pickle
import numpy as np
np.random.seed(42)

print("Loading training data...")

with open('/home/mingdayang/mmdetection3d/data/nuscenes/mmdet3d_bevformer/nuscenes_infos_temporal_train.pkl', 'rb') as f:
    data_train = pickle.load(f)

print("Data loaded!")
print(f'Keys of train data: {data_train.keys()}')
print(f'Number of train samples: {len(data_train["infos"])}')
print(data_train['metadata'])

print("Sampling 1/4 of the training data...")
# Sample 1/4 of train data
n_train_samples = len(data_train['infos']) // 4
sampled_train_indices = np.random.choice(len(data_train['infos']), size=n_train_samples, replace=False)
sampled_train_infos = [data_train['infos'][i] for i in sampled_train_indices]

# Create new sampled dataset
sampled_data_train = {
    'infos': sampled_train_infos,
    'metadata': data_train['metadata']
}


print(f'Original samples: {len(data_train["infos"])}')
print(f'Sampled samples: {len(sampled_train_infos)}')

with open('/home/mingdayang/mmdetection3d/data/nuscenes/mmdet3d_bevformer/nuscenes_annotation_files_custom/sampled_quarter_nuscenes_infos_temporal_train.pkl', 'wb') as f:
    pickle.dump(sampled_data_train, f)

print("Sampled training data saved.")
print("Exiting")