from align import forward_person_transform
from align import inverse_person_transform
from align import load_train_data, align_person

import cv2
import numpy as np

IMAGES_PATH = 'images/'

IMAGES = ['e1700.jpg', 'e1701.jpg', 'e1702.jpg', 'e1703.jpg', 'e1704.jpg']
MASKS = ['e1700_seg_read.png', 'e1701_seg_read.png', 'e1702_seg_read.png', 'e1703_seg_read.png', 'e1704_seg_read.png']

IMAGES_ALL = [IMAGES_PATH + i for i in IMAGES[1:]]
MASKS_ALL = [IMAGES_PATH + i for i in MASKS[1:]]

TEST_IMAGE = IMAGES_PATH + 'img_test.jpg'
TEST_MASK = IMAGES_PATH + 'test_mask_voc_seg_read.jpg'

def test_forward_transform():
    image = cv2.imread(TEST_IMAGE)
    mask = cv2.imread(TEST_MASK, cv2.IMREAD_UNCHANGED)

    mask = np.array(mask > 0, np.uint8)

    person, person_mask = forward_person_transform(image, mask)

    cv2.imwrite('forward_transform_' + TEST_IMAGE.split('/')[-1], person)
    cv2.imwrite('forward_transform_' + TEST_MASK.split('/')[-1], person_mask * 255)

def test_inverse_transform():
    image = cv2.imread(TEST_IMAGE)
    mask = cv2.imread(TEST_MASK, cv2.IMREAD_UNCHANGED)

    mask = np.array(mask > 0, np.uint8)

    person, person_mask = forward_person_transform(image, mask)

    blended_image = inverse_person_transform(image, mask, person, person_mask)

    cv2.imwrite('inverse_person_transform_' + TEST_IMAGE.split('/')[-1], blended_image)


def test_align_person():
    image = cv2.imread(TEST_IMAGE)
    mask = cv2.imread(TEST_MASK, cv2.IMREAD_UNCHANGED)

    mask = np.array(mask > 0, np.uint8)

    person, person_mask = forward_person_transform(image, mask)
    train_images, train_masks = load_train_data(IMAGES_ALL, MASKS_ALL)

    aligned_image, aligned_mask = align_person(person, person_mask[..., None], train_images, train_masks)

    cv2.imwrite('align_transform_' + TEST_IMAGE.split('/')[-1], aligned_image)
    cv2.imwrite('align_transform_' + TEST_MASK.split('/')[-1], aligned_mask * 255)


def test_align_person_partial():
    image = cv2.imread(TEST_IMAGE)
    mask = cv2.imread(TEST_MASK, cv2.IMREAD_UNCHANGED)

    mask = np.array(mask > 0, np.uint8)

    mask[150:] = 0

    person, person_mask = forward_person_transform(image, mask)
    train_images, train_masks = load_train_data(IMAGES_ALL, MASKS_ALL)

    aligned_image, aligned_mask = align_person(person, person_mask[..., None], train_images, train_masks)

    cv2.imwrite('align_transform_partial_' + TEST_IMAGE.split('/')[-1], aligned_image)
    cv2.imwrite('align_transform_partial_' + TEST_MASK.split('/')[-1], aligned_mask * 255)


test_forward_transform()
test_inverse_transform()
test_align_person()
test_align_person_partial()
