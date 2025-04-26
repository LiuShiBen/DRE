import numpy as np
import torch
import torch.nn.functional as F
from pkd.utils import CatMeter, make_dirs
from pkd.evaluation.metric import tensor_cosine_dist, tensor_euclidean_dist
from .visualising_rank import visualize_ranked_results
from einops import rearrange
import os
import os.path as osp
import cv2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# def visualize(config, base, loaders):
#
# 	base.set_eval()
#
# 	# meters
# 	query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
# 	gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()
#
# 	# init dataset
# 	if config.visualize_dataset == 'market':
# 		_datasets = [loaders.market_query_samples, loaders.market_gallery_samples]
# 		_loaders = [loaders.market_query_loader, loaders.market_gallery_loader]
# 	elif config.visualize_dataset == 'duke':
# 		_datasets = [loaders.duke_query_samples, loaders.duke_gallery_samples]
# 		_loaders = [loaders.duke_query_loader, loaders.duke_gallery_loader]
# 	elif config.visualize_dataset == 'customed':
# 		_datasets = [loaders.query_samples, loaders.gallery_samples]
# 		_loaders = [loaders.query_loader, loaders.gallery_loader]
#
# 	# compute query and gallery features
# 	with torch.no_grad():
# 		for loader_id, loader in enumerate(_loaders):
# 			for data in loader:
# 				# compute feautres
# 				images, pids, cids = data
# 				images = images.to(base.device)
# 				features = base.model(images)
# 				# save as query features
# 				if loader_id == 0:
# 					query_features_meter.update(features.data)
# 					query_pids_meter.update(pids)
# 					query_cids_meter.update(cids)
# 				# save as gallery features
# 				elif loader_id == 1:
# 					gallery_features_meter.update(features.data)
# 					gallery_pids_meter.update(pids)
# 					gallery_cids_meter.update(cids)
#
# 	# compute distance
# 	query_features = query_features_meter.get_val()
# 	gallery_features = gallery_features_meter.get_val()
#
# 	if config.test_metric is 'cosine':
# 		distance = tensor_cosine_dist(query_features, gallery_features).data.cpu().numpy()
#
# 	elif config.test_metric is 'euclidean':
# 		distance = tensor_euclidean_dist(query_features, gallery_features).data.cpu().numpy()
#
# 	# visualize
# 	visualize_ranked_results(distance, _datasets, config.visualize_output_path, mode=config.visualize_mode, only_show=config.visualize_mode_onlyshow)


# visualize sampled patch
def featuremaps2heatmaps(base, imgs, theta, image_paths, current_step, current_epoch, if_save=False, grid_size=16):
    _, _, height, width = imgs.shape
    k = width // grid_size
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 0, 255)]

    grid_img_tensor = []
    if if_save:
        save_dir = osp.join(base.output_dirs_dict['images'], str(current_step) + '_' + str(current_epoch))
        make_dirs(save_dir)
    for j in range(imgs.size(0)):
        # get image name
        path = image_paths[j]
        imname = osp.basename(osp.splitext(path)[0])

        # RGB image
        img = imgs[j, ...]
        for t, m, s in zip(img, IMAGENET_MEAN, IMAGENET_STD):
            t.mul_(s).add_(m).clamp_(0, 1)
        img_np = np.uint8(np.floor(img.numpy() * 255)).transpose((1, 2, 0)).copy()  # (c, h, w) -> (h, w, c)

        # draw patch bounding box
        for i in range(theta.shape[1]):
            loc = int(theta[j, i, :, :].argmax())
            h, w = loc // k, loc % k
            cv2.rectangle(img_np, (w * grid_size, h * grid_size), ((w + 1) * grid_size, (h + 1) * grid_size), colors[i],
                          2)

        # save images in a single figure (add white spacing between images)
        # from left to right: original image, activation map, overlapped image
        grid_img_tensor.append(img_np)
        if if_save:
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), img_np[:, :, ::-1])
    grid_img_tensor = np.transpose(np.stack(grid_img_tensor, axis=0), (0, 3, 1, 2))
    return torch.from_numpy(grid_img_tensor)


def visualize(config, base, loaders):
    base.set_all_model_eval()

    with torch.no_grad():
        for dataset_name, temp_loaders in loaders.test_loader_dict.items():
            if dataset_name != 'cuhk03':
                continue
            for data in temp_loaders[0]:
                # compute feautres
                images, pids, cids = data[0:3]
                image_paths = data[5]
                images = images.to(base.device)
                features, _, feature_maps = base.model_dict['tasknet'](images, [], force_output_map=True)
                theta = base.model_dict['patchnet'](feature_maps)
                featuremaps2heatmaps(base, images.cpu().float(), theta.cpu().float(), image_paths,
                                     4, 50, if_save=True)
                break


# # save patch feature
# def generate_patch_features(x, theta):
#     output = (x[:, np.newaxis] * theta[:, :, np.newaxis]).sum(dim=(3, 4))
#     return rearrange(output, 'n k c -> (n k) c')
#
#
#
# def visualize(config, base, loaders):
#     base.set_all_model_eval()
#
#     feature_dict = {}
#
#     with torch.no_grad():
#         for dataset_name, temp_loaders in loaders.test_loader_dict.items():
#             for data in temp_loaders[0]:
#                 # compute feautres
#                 images, pids, cids = data[0:3]
#                 images = images.to(base.device)
#                 features, _, feature_maps = base.model_dict['tasknet'](images, [], force_output_map=True)
#                 theta = base.model_dict['patchnet'](feature_maps)
#                 patch_features = generate_patch_features(feature_maps, theta)
#                 feature_dict[dataset_name] = features.cpu().numpy()
#                 feature_dict[dataset_name + '_patch'] = patch_features.cpu().numpy()
#                 break
#     np.savez('temp', **feature_dict)
