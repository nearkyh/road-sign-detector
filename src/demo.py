import argparse
import cv2
import torch

from model import RoadSignModel
from data import preprocess
from utils import draw_bbox


class Demo:

    def __init__(self, video_file, weights_file):
        self.video_file = video_file
        self.weights_file = weights_file

        self.cap = cv2.VideoCapture(video_file)

    def detect(self, img):
        model = RoadSignModel()
        model.load_state_dict(torch.load(
            f=self.weights_file,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ))
        model.eval()
        with torch.no_grad():
            img = preprocess(img)
            out_cls, out_bbox = model(img)

        return out_cls, out_bbox

    def run(self):
        while True:
            ret, img = self.cap.read()
            if ret:
                img_h, img_w, _ = img.shape
                out_cls, out_bbox = self.detect(img)
                print(out_cls, out_bbox)
                # print(out_cls, out_bbox)
                # draw_bbox(img, out_bbox)

            else:
                break

            cv2.imshow('Road Sign Detector', img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='Enter the video file path.')
    parser.add_argument('--weights_file', type=str, help='Enter the weights file path.')
    args = parser.parse_args()
    video_file = args.video_file
    weights_file = args.weights_file

    if (video_file is not None) and (weights_file is not None):
        demo = Demo(
            video_file=video_file,
            weights_file=weights_file
        )
        demo.run()
    else:
        parser.print_help()