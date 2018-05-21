from params import *
from align import *

import numpy as np
import cv2


# predictions blending methods
def masked_images_blending(img1, img1_mask, img2, img2_mask, alpha):
    mask_intersection = img1_mask * img2_mask

    return img1 * (img1_mask - mask_intersection) \
         + img2 * (img2_mask - mask_intersection) \
         + alpha * img1 * mask_intersection \
         + (1 - alpha) * img2 * mask_intersection


def blend_background(initial_image, prediction, mask, morph_param, alpha):
    background_image = initial_image.copy()

    kernel = np.ones(morph_param, np.uint8)
    background_mask = cv2.erode(mask, kernel, iterations = 1)
    background_mask = 1 - background_mask

    return masked_images_blending(background_image, background_mask[..., None], prediction, mask, alpha)


def forward_person_transform(image, mask):
    x, y, w, h = cv2.boundingRect(mask)

    mask_cumsum = np.cumsum(mask.sum(axis=0))
    total_sum = mask_cumsum[-1]
    cummask = np.zeros_like(mask_cumsum)
    cummask[mask_cumsum < total_sum * (1 - CUM_REDUCE)] = 1
    cummask[mask_cumsum < total_sum * CUM_REDUCE] = 0
    cummask = cummask > 0

    masked = np.where(cummask)[0]
    masked_l = max(0, masked.min() - CUM_INCREASE * w)
    masked_r = min(image.shape[1], masked.max() + CUM_INCREASE * w)

    cum_w = (masked_r - masked_l)
    print(cum_w / h)
    if cum_w / h > 0.3:
        fx = MEAN_WIDTH / cum_w
    else:
        fx = MEAN_WIDTH / cum_w

    fx = IMAGE_H / h

    person = cv2.resize(image[y:y+h, x:x+w], None, fx=fx, fy=fx)
    person_mask = cv2.resize(mask[y:y+h, x:x+w], None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)

    person_y = person.shape[0]
    if person_y > IMAGE_H - 2 * HEAD_Y:
        fy = (IMAGE_H - 2 * HEAD_Y) / person_y
        person = cv2.resize(person, None, fx=fy, fy=fy)
        person_mask = cv2.resize(person_mask, None, fx=fy, fy=fy, interpolation=cv2.INTER_NEAREST)

    return person, person_mask


def inverse_person_transform(image, image_mask, prediction, prediction_mask):
    ix, iy, iw, ih = cv2.boundingRect(image_mask)
    px, py, pw, ph = cv2.boundingRect(prediction_mask)

    prediction_inverse = np.zeros_like(image)
    prediction_inverse[iy:iy+ih, ix:ix+iw] = cv2.resize(prediction[py:py+ph, px:px+pw], (iw, ih))

    return blend_background(image, prediction_inverse, image_mask[..., None], MORPH, ALPHA)

from glob import glob

def load_train_data(images_path):
#    assert len(images_paths) == len(masks_paths), "wrong size, images length should be equal to masks length"
    images_paths = glob(images_path + '*.png')
#    print(images_paths)
    train_images = np.zeros((len(images_paths), IMAGE_H, IMAGE_W, 3), np.uint8)
    train_masks = np.zeros((len(images_paths), IMAGE_H, IMAGE_W, 1), np.uint8)

    for i in range(len(images_paths)):
        train_images[i] = cv2.imread(images_paths[i], cv2.IMREAD_UNCHANGED)[..., :3]
        train_masks[i] = cv2.imread(images_paths[i], cv2.IMREAD_UNCHANGED)[..., 3:] > 0

    return train_images, train_masks


def inpaint_image(image, mask, y):
    mx, my, mw, mh = cv2.boundingRect(mask)

    from_x = max(0, mx - W_INCREASE)
    to_x = min(image.shape[1], mx + mw + W_INCREASE)
    to_y = min(my + mh, y)

    inpaint_y = to_y
    if inpaint_y < 180:
        inpaint_y -= INPAINT_INCREASE

    print(to_y, inpaint_y)

    inpainting_mask = np.zeros((inpaint_y, image.shape[1]), np.uint8)
    inpainting_mask[:inpaint_y, from_x:to_x] = 1

    result = image.copy()
    result[:inpaint_y] = cv2.inpaint(image[:inpaint_y], inpainting_mask, 3, cv2.INPAINT_TELEA)

    return result


# put image into fake background and/or add parts from best fit person if we have partial image
def align_person(image, mask, train_images, train_masks):
    partial_image = np.zeros(MODEL_INPUT_SIZE)
    partial_mask = np.zeros(MASK_SIZE)

    head_w = image.shape[1]
    head_x = max(0, (IMAGE_W - head_w) // 2)

    partial_image[HEAD_Y:HEAD_Y + image.shape[0], head_x:head_x + head_w] = image * mask
    partial_mask[HEAD_Y:HEAD_Y + image.shape[0], head_x:head_x + head_w] = mask

    # get best base image
    masks_intersection = np.sum(train_masks * partial_mask, axis=(1,2,3))
    masks_union = np.sum((train_masks + partial_mask) > 0, axis=(1,2,3))
    best_fit = np.argmax(masks_intersection / masks_union)

    base_image = inpaint_image(train_images[best_fit], train_masks[best_fit][..., 0].astype(np.uint8), HEAD_Y + image.shape[0])

    blended_aligned = blend_background(base_image, partial_image, partial_mask, MORPH, ALPHA)
    return blended_aligned, partial_mask


