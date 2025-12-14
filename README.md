# BoT-Seg

This project investigates whether instance segmentation can provide measurable benefits to multi-object tracking (MOT) pipelines when integrated in a conservative, detection-assist role. Using BoT-SORT with a YOLOX detector as a strong baseline, we introduce a segmentation-assisted refinement module that leverages YOLOv8-SEG masks to adjust detection confidence prior to data association. The proposed approach is explicitly designed to be non-invasive: segmentation is never used to create new tracks, resize bounding boxes, or override detector outputs, but only to softly reinforce detections when spatial agreement is observed.
    
Experiments are conducted on a dense pedestrian video. Tracking performance is evaluated using proxy MOT metrics, including average active tracks per frame, total unique track identities, track length distributions, and estimated ID switches. Results show that the segmentation-assisted variant produces tracking behavior nearly identical to the baseline BoT-SORT pipeline, with no consistent improvement in track stability or identity preservation. These findings suggest that naïve segmentation integration, particularly when using models trained on generic datasets such as COCO, provides limited benefit for modern tracking systems already dominated by strong detection and association mechanisms.

# Demo Video Download
https://drive.google.com/drive/folders/1U2Z21qJxRjsBYjXpVZpLOPXU3BE0BUKk?usp=sharing
