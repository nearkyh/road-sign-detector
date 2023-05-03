import cv2
import numpy as np


def read_img(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def create_bbox_mask(img, bbox):
    """이미지의 Bounding Box에 대한 Mask 생성"""
    img_h, img_w, _ = img.shape
    mask = np.zeros((img_h, img_w))
    bbox = np.array(bbox, dtype=np.int32)
    xmin, ymin, xmax, ymax = bbox
    mask[ymin: ymax, xmin: xmax] = 1.

    return mask


def mask_to_bbox(mask):
    """Mask에 대한 Bounding Box 반환"""
    y_arr, x_arr = np.nonzero(mask)  # 0이 아닌 index(y, x)들을 반환
    if len(x_arr) == 0:
        bbox = np.zeros(4, dtype=np.float32)
    else:
        xmin = np.min(x_arr)
        ymin = np.min(y_arr)
        xmax = np.max(x_arr)
        ymax = np.max(y_arr)
        bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    return bbox


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """Bounding Box 그리기"""
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)


if __name__ == '__main__':
    img = np.zeros((300, 300, 3), np.uint8)
    bbox = [35, 35, 75, 75]
    draw_bbox(img, bbox)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()