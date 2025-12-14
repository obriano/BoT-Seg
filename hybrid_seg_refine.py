import numpy as np

def refine_yolox_dets_with_seg(
    yolox_dets,
    seg_result,
    img_shape,
    seg_alpha=0.8,
    min_seg_iou=0.5,
    min_seg_fill=0.1,
    only_person=True,
):
    """
    Args:
        yolox_dets: np.ndarray of shape (N, 7):
            [x1, y1, x2, y2, score, cls_conf, cls_id]
        seg_result: a single Ultralytics Result from YOLOv8-SEG
        img_shape: H, W, C of the original image (unused except for sanity)
        seg_alpha: how strongly to boost scores based on mask quality
                   (0.0 = no boost, 1.0 = strong boost)
        min_seg_iou: minimum IoU between YOLOX box and YOLOv8 box to trust a mask
        min_seg_fill: minimum fraction of the YOLOX box area that must be covered
                      by the mask to consider it "supporting" the detection
        only_person: if True, only apply segmentation refinement to cls_id == 0

    Returns:
        refined_dets: np.ndarray, same shape as yolox_dets, with scores possibly
                      boosted (column 4). Everything else identical.
    """
    if yolox_dets is None or len(yolox_dets) == 0:
        return yolox_dets

    dets = yolox_dets.copy()

    if seg_result is None or seg_result.masks is None or seg_result.masks.data is None:
        return dets

    seg_boxes = seg_result.boxes.xyxy.cpu().numpy().astype(np.float32)  # (M,4)
    seg_masks = seg_result.masks.data.cpu().numpy().astype(np.float32)  # (M,Hm,Wm)
    Hm, Wm = seg_masks.shape[1], seg_masks.shape[2]

    def iou_one_to_many(box, boxes):
        if boxes.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h

        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = np.maximum(1e-6, area_a + area_b - inter)
        return inter / union

    H, W = img_shape[:2]

    for i in range(dets.shape[0]):
        x1, y1, x2, y2, score, cls_conf, cls_id = dets[i]

        if only_person and int(cls_id) != 0:
            continue

        bx1 = max(0, min(W - 1, x1))
        by1 = max(0, min(H - 1, y1))
        bx2 = max(0, min(W, x2))
        by2 = max(0, min(H, y2))
        if bx2 <= bx1 or by2 <= by1:
            continue

        ious = iou_one_to_many(np.array([bx1, by1, bx2, by2], dtype=np.float32), seg_boxes)
        if ious.shape[0] == 0:
            continue

        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])

        if best_iou < min_seg_iou:
            continue

        mask = seg_masks[best_idx]  # (Hm, Wm)

        scale_x = Wm / float(W)
        scale_y = Hm / float(H)
        mx1 = int(max(0, min(Wm - 1, bx1 * scale_x)))
        my1 = int(max(0, min(Hm - 1, by1 * scale_y)))
        mx2 = int(max(0, min(Wm, bx2 * scale_x)))
        my2 = int(max(0, min(Hm, by2 * scale_y)))
        if mx2 <= mx1 or my2 <= my1:
            continue

        box_mask = mask[my1:my2, mx1:mx2]
        if box_mask.size == 0:
            continue

        fill = float(box_mask.mean())  # [0,1]

        if fill < min_seg_fill:
            continue

        boost_factor = 1.0 + seg_alpha * fill

        boosted_score = score * boost_factor

        dets[i, 4] = max(score, boosted_score)

    return dets
