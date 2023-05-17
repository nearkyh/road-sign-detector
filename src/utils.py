import cv2
import numpy as np

class_label = {
    0: 'speedlimit',
    1: 'stop',
    2: 'crosswalk',
    3: 'trafficlight',
}
class_color = {
    0: (0, 255, 0),
    1: (0, 255, 0),
    2: (0, 255, 0),
    3: (0, 255, 0),
}


def draw_objects(img, bboxes, cls_ids):
    for idx in range(len(bboxes)):
        xmin, ymin, xmax, ymax = bboxes[idx]
        cls_id = cls_ids[idx]
        cls_label = class_label[cls_id]
        bbox_color = class_color[cls_id]
        bbox_thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_color = bbox_color
        font_thickness = bbox_thickness
        (font_w, font_h), _ = cv2.getTextSize(cls_label, font, font_size, font_thickness)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bbox_color, bbox_thickness)
        cv2.putText(img, cls_label, (xmin, ymin + font_h), font, font_size, font_color, font_thickness)


if __name__ == '__main__':
    img = np.zeros((1000, 1000, 3), np.uint8)
    bboxes = [[35, 35, 575, 575]]
    cls_ids = [1]
    draw_objects(img, bboxes, cls_ids)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()