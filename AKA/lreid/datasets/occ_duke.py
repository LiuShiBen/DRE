#from __future__ import division, print_function, absolute_import
import os
import copy
import os.path as osp
from lreid.data_loader.incremental_datasets import IncrementalPersonReIDSamples
from lreid.data.datasets import ImageDataset
import re
import glob

class IncrementalSamples4occduke(IncrementalPersonReIDSamples):
    '''
    Duke dataset
    '''
    duke_path = 'Occluded_Duke'
    def __init__(self, datasets_root, relabel=True, combineall=False):
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root, self.duke_path)
        self.train_dir = osp.join(
            root, 'bounding_box_train'
        )
        self.query_dir = osp.join(root, 'query')
        self.gallery_dir = osp.join(
            root, 'bounding_box_test'
        )

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)

    def process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append([img_path, pid, camid, 'occduke', pid])

        return data

