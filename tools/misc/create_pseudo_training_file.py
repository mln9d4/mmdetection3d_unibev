import mmengine
from nuscenes.utils.data_classes import Box
import numpy as np

# 1. Load your prediction results and the original info file
results = mmengine.load('/home/mingdayang/mmdetection3d/outputs/inference/baseline/unibev_C_full_nuscenes/pts_bbox/results_nusc.json')
infos = mmengine.load('/home/mingdayang/mmdetection3d/data/nuscenes/mmdet3d_bevformer/nuscenes_infos_temporal_val.pkl')

# 2. Reorganize results by sample_token
pred_dict = {}
for det in results['results'].values():
    # results['results'] is a dict where keys are sample_tokens
    pass 
# Note: NuScenes result format already groups by sample_token

# 3. Update Infos
new_infos = []
for info in infos['data_list']:
    token = info['token']
    if token in results['results']:
        preds = results['results'][token]
        
        # Filter and extract boxes/names/velocity
        # Convert Global coords back to LiDAR if necessary
        gt_boxes_3d = []
        gt_names = []
        
        for p in preds:
            if p['score'] > 0.4: # Confidence Threshold
                # [x, y, z, w, l, h, yaw]
                box = p['translation'] + p['size'] + [p['rotation']] 
                gt_boxes_3d.append(box)
                gt_names.append(p['detection_name'])
        
        info['gt_boxes'] = np.array(gt_boxes_3d)
        info['gt_names'] = np.array(gt_names)
    
    new_infos.append(info)

# 4. Save as a new pseudo-label info file
mmengine.dump({'data_list': new_infos, 'metainfo': infos['metainfo']}, 'nuscenes_infos_train_pseudo.pkl')