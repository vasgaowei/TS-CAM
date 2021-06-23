import os
import cv2
import numpy as np
import pickle
import torch
from utils import mkdir


def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam


def blend_cam(image, cam):
    cam = (cam * 255.).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blend = image * 0.5 + heatmap * 0.5

    return blend, heatmap


def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)

    try:
        _, contours, _ = cv2.findContours(thr_gray_heatmap,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]

    return estimated_bbox


def tensor2image(input, image_mean, image_std):
    image_mean = torch.reshape(torch.tensor(image_mean), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(image_std), (1, 3, 1, 1))
    image = input * image_mean + image_std
    image = image.numpy().transpose(0, 2, 3, 1)
    image = image[:, :, :, ::-1] * 255
    return image


def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def draw_bbox(image, iou, gt_box, pred_box, gt_score, is_top1=False):

    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color1, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color2, 2)
        return img

    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box, pred_box)

    # mark the iou
    mark_target(boxed_image, '%.1f' % (iou * 100), (140, 30), 2)
    # mark_target(boxed_image, 'IOU%.2f' % (iou), (80, 30), 2)
    # # mark the top1
    # if is_top1:
    #     mark_target(boxed_image, 'Top1', (10, 30))
    # mark_target(boxed_image, 'GT_Score%.2f' % (gt_score), (10, 200), 2)

    return boxed_image


def evaluate_cls_loc(input, cls_label, bbox_label, logits, cams, image_names, cfg, epoch):
    """
    :param input: input tensors of the model
    :param cls_label: class label
    :param bbox_label: bounding box label
    :param logits: classification scores
    :param cams: cam of all the classes
    :param image_names: names of images
    :param cfg: configurations
    :param epoch: epoch
    :return: evaluate results
    """
    cls_top1 = []
    cls_top5 = []
    loc_top1 = []
    loc_top5 = []
    loc_gt_known = []

    # label, top1 and top5 results
    cls_label = cls_label.tolist()
    cls_scores = logits.tolist()
    _, top1_idx = logits.topk(1, 1, True, True)
    top1_idx = top1_idx.tolist()
    _, top5_idx = logits.topk(5, 1, True, True)
    top5_idx = top5_idx.tolist()

    k = cfg.MODEL.TOP_K
    _, topk_idx = logits.topk(k, 1, True, True)
    topk_idx = topk_idx.tolist()

    batch = cams.shape[0]
    image = tensor2image(input.clone().detach().cpu(), cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)

    for b in range(batch):

        # mean top k
        cam_b = cams[b, [cls_label[b]], :, :]
        cam_b = torch.mean(cam_b, dim=0, keepdim=True)

        cam_b = cam_b.detach().cpu().numpy().transpose(1, 2, 0)

        # Resize and Normalize CAM
        cam_b = resize_cam(cam_b, size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
        # Estimate BBOX
        estimated_bbox = get_bboxes(cam_b, cam_thr=cfg.MODEL.CAM_THR)

        # Calculate IoU
        iou = calculate_IOU(bbox_label[b].numpy(), estimated_bbox)

        # top1
        gt_score = cls_scores[b][top1_idx[b][0]]  # score of gt class
        if cls_label[b] in top1_idx[b]:
            is_top1 = True
            cls_top1.append(1)
            if iou>=0.5:
                loc_top1.append(1)
            else:
                loc_top1.append(0)
        else:
            is_top1 = False
            cls_top1.append(0)
            loc_top1.append(0)

        # top5
        if cls_label[b] in top5_idx[b]:
            cls_top5.append(1)
            if iou>=0.5:
                loc_top5.append(1)
            else:
                loc_top5.append(0)
        else:
            cls_top5.append(0)
            loc_top5.append(0)

        # gt known
        if iou >= 0.5:
            loc_gt_known.append(1)
        else:
            loc_gt_known.append(0)

        # Get blended image
        blend, heatmap = blend_cam(image[b], cam_b)
        # Get boxed image
        boxed_image = draw_bbox(blend, iou, bbox_label[b].numpy(), estimated_bbox, gt_score, is_top1)

        # save result
        if cfg.TEST.SAVE_BOXED_IMAGE:
            image_name = image_names[b]

            save_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'boxed_image', str(epoch), image_name.split('/')[0])
            save_path = os.path.join(cfg.BASIC.SAVE_DIR, 'boxed_image', str(epoch), image_name)
            mkdir(save_dir)
            # print(save_path)
            cv2.imwrite(save_path, boxed_image)

    return cls_top1, cls_top5, loc_top1, loc_top5, loc_gt_known


