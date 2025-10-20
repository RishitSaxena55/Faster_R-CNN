import xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim

# -------------- Data Untils -------------------

def parse_annotation(annotation_path, image_dir, img_size):
    '''
    Traverse the xml tree, get the annotations, and resize them to the scaled image size
    '''
    img_h, img_w = img_size

    with open(annotation_path, "r") as f:
        tree = ET.parse(f)

    root = tree.getroot()  
    
    img_paths = []
    gt_boxes_all = []
    gt_classes_all = []
    # get image paths
    for object_ in root.findall('image'):
        img_path = os.path.join(image_dir, object_.get("name"))
        img_paths.append(img_path)
      
        # get raw image size    
        orig_w = int(object_.get("width"))
        orig_h = int(object_.get("height"))
            
        # get bboxes and their labels   
        groundtruth_boxes = []
        groundtruth_classes = []
        for box_ in object_.findall('box'):
            xmin = float(box_.get("xtl"))
            ymin = float(box_.get("ytl"))
            xmax = float(box_.get("xbr"))
            ymax = float(box_.get("ybr"))
        
            # rescale bboxes
            bbox = torch.Tensor([xmin, ymin, xmax, ymax])
            bbox[[0, 2]] = bbox[[0, 2]] * img_w/orig_w
            bbox[[1, 3]] = bbox[[1, 3]] * img_h/orig_h
        
            groundtruth_boxes.append(bbox.tolist())

            # get labels
            label = box_.get("label")
            groundtruth_classes.append(label)

        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(groundtruth_classes)
                
    return gt_boxes_all, gt_classes_all, img_paths

def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
    
    return fig, axes

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1
        
    return fig, ax

def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax

def gen_anc_centers(out_size):
    out_h, out_w = out_size

    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5

    return anc_pts_x, anc_pts_y

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)

    # shape - [1, Wmap, Hmap, n_anchor_boxes, 4]
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0), anc_pts_y.size(dim=0), n_anc_boxes, 4)

    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale

                    xmin = xc - w / 2
                    xmax = xc + w / 2
                    ymin = yc - h / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1
            
            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)

    return anc_base

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']
    
    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes
    
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor
        
    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)
    
    return proj_bboxes

def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)
    
    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))
    
    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)
    
    return ious_mat

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert( pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx -  anc_cx) / anc_w
    ty_ = (gt_cy - anc_cy) / anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    """
    Args:
        anc_boxes_all: [B, w_amap, h_amap, n_anc_boxes, 4]
        gt_bboxes_all: [B, max_objects, 4]
        gt_classes_all: [B, max_objects]
    """
    batch_size, max_objects = gt_bboxes_all.shape[0], gt_bboxes_all.shape[1]
    n_anc_boxes = anc_boxes_all.shape[3]
    # Compute IoU matrix
    iou_mat = get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all)

    # Compute max IoU per gt_bbox (also this is condition-1 for positive sampling)
    max_iou_per_gt, _ = torch.max(iou_mat, dim=1, keepdim=True)
    pos_anc_mask_c1 = torch.where(iou_mat == max_iou_per_gt, 1, 0)

    # Condition-2 select anc_boxes with IoU > pos_thresh with all gt_bboxes
    pos_anc_mask_c2 = torch.where(iou_mat > pos_thresh, 1, 0)

    # Combine the two conditions by Logical OR
    pos_anc_mask = torch.logical_or(pos_anc_mask_c1, pos_anc_mask_c2)

    # As no. of pos samples may be diff for diff images, so flatten the batch
    pos_anc_mask = torch.flatten(pos_anc_mask, start_dim=0, end_dim=1)
    
    pos_anc_ind = torch.nonzero(pos_anc_mask, as_tuple=True)[0]
    pos_anc_iou = torch.gather(iou_mat.flatten(start_dim=0, end_dim=-2), index=pos_anc_ind.unsqueeze(dim=0)).squeeze()
    
    # Computing offsets
    # We have to map +ve samples to gt_bbox
    # Expand gt_bboxes
    gt_bboxes_exp = gt_bboxes_all.expand(-1, n_anc_boxes, max_objects, 4)

    # For each anc_box select gt_box with highest IoU
    _, max_iou_per_anc_ind = torch.max(iou_mat, dim=2, keepdim=True)

    # Gather the gt_bboxes
    # usqueeze max_iou_per_anc_ind [B, 1, max_objects] -> [B, 1, max_objects, 1]
    # As torch.gather expects same no. of dimensions
    max_iou_per_anc_ind = torch.unsqueeze(max_iou_per_anc_ind, dim=-1)
    max_iou_per_anc_ind_exp = max_iou_per_anc_ind.expand(-1, -1, -1, 4)

    mapped_gt_bboxes = torch.gather(gt_bboxes_exp, dim=-2, index=max_iou_per_anc_ind_exp)

    # Flatten the batches
    mapped_gt_bboxes = torch.flatten(mapped_gt_bboxes, start_dim=0, end_dim=-2)
    # Expand pos_anc_ind to match dim
    pos_anc_ind_exp = pos_anc_ind.expand(-1, 4)
    # Gather the pos samples
    gt_bbox_mapping = torch.gather(mapped_gt_bboxes, pos_anc_ind_exp, dim=0)

    # Assign categories to pos samples using same process
    # Expand gt_classes_all to n_anc_boxes
    gt_classes_all_exp = gt_classes_all.expand(-1, n_anc_boxes, max_objects)
    # Gather the classes
    cl_anc = torch.gather(gt_classes_all_exp, index=max_iou_per_anc_ind, dim=-1)
    # Gather pos sample classes
    pos_anc_ind_exp_c = pos_anc_ind.expand(-1, 1)
    pos_cl_anc = torch.gather(cl_anc, index=pos_anc_ind_exp_c, dim=0)

    # Flatten anc_boxes_all
    anc_boxes_all = torch.flatten(anc_boxes_all, start_dim=0, end_dim=-2)
    # Gather pos_anc_coords
    pos_anc_coords = anc_boxes_all[pos_anc_ind]
    # Calculate offsets
    offsets = calc_gt_offsets(pos_anc_coords, gt_bbox_mapping)

    # Sample -ve samples
    # neg_anc are samples whose IoU < neg_thresh 
    neg_anc_mask = torch.where(iou_mat < neg_thresh, 1, 0)
    # Flatten batches
    neg_anc_mask = torch.flatten(neg_anc_mask, start_dim=0, end_dim=-2)
    # Find indices
    neg_anc_ind = torch.nonzero(neg_anc_mask, as_tuple=True)[0]

    # Random sample -ve anc boxes to match +ve amc boxes
    neg_anc_ind = neg_anc_ind[torch.randint(0, neg_anc_ind.shape[0], (pos_anc_ind.shape[0],))]
    neg_anc_coords = anc_boxes_all[neg_anc_ind]

    return pos_anc_ind, neg_anc_ind, pos_anc_iou, offsets, pos_cl_anc, pos_anc_coords, neg_anc_coords, pos_anc_ind



    











    






