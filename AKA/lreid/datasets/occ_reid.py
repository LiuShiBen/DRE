#from __future__ import division, print_function, absolute_import
import os
import copy
import os.path as osp
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
from lreid.data.datasets import ImageDataset
import re
import glob

import glob

class IncrementalSamples4occreid(IncrementalPersonReIDSamples):
    '''
    Duke dataset
    '''
    occ_path = 'Occluded_REID'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.occ_path)
        #self.train_dir = osp.join(root, 'bounding_box_train')
        self.query_dir = osp.join(root, 'occluded_body_images')
        self.gallery_dir = osp.join(root, 'whole_body_images')

        train = []
        query = self.process_dir(self.query_dir, relabel=False)  # occluded_body_images
        gallery = self.process_dir(self.gallery_dir, relabel=False, is_query=False)  # whole_body_images
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.tif'))  # 原join(dir_path,'*','*.jpg')
        if is_query:
            camid = 0
        else:
            camid = 1
        pid_container = set()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, 'occreid', pid))
        return data
