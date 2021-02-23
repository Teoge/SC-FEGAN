import argparse
import cv2
import numpy as np
from utils.config import Config
from model import Model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help="Image path.")
    parser.add_argument('--mask', required=True, help="Mask path.")
    return parser


config = Config('demo.yaml')
model = Model(config)
model.load_demo_graph(config)


def make_noise():
    noise = np.zeros([512, 512, 1], dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = np.expand_dims(noise, axis=0)
    return noise


def make_sketch():
    sketch = np.zeros((512, 512, 3))
    sketch = np.asarray(sketch[:, :, 0]/255, dtype=np.uint8)
    sketch = np.expand_dims(sketch, axis=2)
    return np.expand_dims(sketch, axis=0)


def make_stroke():
    stroke = np.zeros((512, 512, 3))
    stroke = stroke/127.5 - 1
    return np.expand_dims(stroke, axis=0)


def make_mask(mask):
    # mask = np.zeros((512, 512, 3))
    # for pt in pts:
    #     cv2.line(mask, pt['prev'], pt['curr'], (255, 255, 255), 12)
    mask = np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
    mask = np.expand_dims(mask, axis=2)
    return np.expand_dims(mask, axis=0)


def preprocess(image):
    image = image / 127.5 - 1
    return np.expand_dims(image, axis=0)


def postprocess(result):
    result = (result + 1) * 127.5
    return np.asarray(result[0, :, :, :], dtype=np.uint8)


def inpaint_face(image, mask):
    sketch = make_sketch()
    stroke = make_stroke()
    noise = make_noise()
    mask = make_mask(mask)

    sketch = sketch * mask
    stroke = stroke * mask
    noise = noise * mask

    image = preprocess(image)
    batch = np.concatenate([image, sketch, stroke, mask, noise], axis=3)
    return postprocess(self.model.demo(config, batch))


if __name__ == '__main__':
    args = get_parser().parse()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    result = inpaint_face(image, mask)
    cv2.imwrite('result.jpg', result)
