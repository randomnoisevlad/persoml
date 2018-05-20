

MEAN_WIDTH
HEAD_ZERO_X
HEAD_ZERO_Y

W_INCREASE

MODEL_INPUT_SIZE
MASK_SIZE

MORPH = 5
ALPHA = 0.5

CUM_REDUCE = 0.1
CUM_INCREASE = 10

# predictions blending methods
def masked_images_blending(img1, img1_mask, img2, img2_mask, alpha):
    mask_intersection = img1_mask * img2_mask
    return img1 * (img1_mask - mask_intersection) \
         + img2 * (img2_mask - mask_intersection) \
         + alpha * img1 * mask_intersection \
         + (1 - alpha) * img2 * mask_intersection


def blend_background(initial_image, prediction, mask, morph_param, alpha):
    background_image = np.zeros_like(initial_image)

    kernel = np.ones(morph_param, np.uint8)
    background_mask = cv2.erode(mask, kernel, iterations = 1)
    background_mask = 1 - background_mask

    return masked_images_blending(background_image, background_mask, prediction, mask, alpha)


def get_image_mask(image):
    return pspnet.get_mask(image)

def get_person_prediction(image):
    return cyclenet.get_prediction(image)

def forward_person_transform(image, mask):
    x, y, w, h = cv2.boudnigBox(mask)

    cummask = np.cumsum(mask.sum(axis=1))
    total_sum = cummask[-1]
    cummask[cummask < total_sum * CUM_REDUCE] = 0
    cummask[cummask > tutal_sum * (1 - CUM_REDUCE)] = 0
    cummask = cummask > 0

    masked = np.where(cummask)[0]
    masked_l = max(0, masked.min() - CUM_INCREASE)
    masked_r = min(image.shape[1], masked.max() + CUM_INCREASE)

    w = masked_r - masked_l
    fx = MEAN_WIDTH / w

    person = cv2.resize(image[x:x+w, y:y+h], fx=fx)
    person_mask = cv2.resize(mask[x:x+w, y:y+h], fx=fx, interpolation=cv2.INTER_NEAREST)

    return person, person_mask


def inverse_person_transform(image, image_mask, prediction, prediction_mask):
    ix, iy, iw, ih = cv2.boudnigBox(image_mask)
    px, py, pw, ph = cv2.boudnigBox(prediction_mask)

    prediction = np.zeros_like(image)
    prediction[ix:ix+iw, iy:iy+ih] = cv2.resize(prediction[px:px+pw, py:py+ph], (iw, ih))

    return blend_background(image, prediction, image_mask, MORPH, ALPHA)


def inpaint_image(image, mask, y):
    mx, my, mw, mh = cv2.boudnigBox(mask)

    from_x = max(0, mx - W_INCREASE)
    to_x = min(image.shape[1], mx + W_INCREASE)
    to_y = min(my, y)

    inpainting_mask = np.zeros((to_y, image.shape[1]))
    inpainting_mask[:to_y, from_x:to_x] = 1

    result = image.copy()
    result[:to_y] = cv2.inpaint(image[:to_y], inpainting_mask, 3, cv2.INPAINT_TELEA)

    return result


# put image into fake background and/or add parts from best fit person if we have partial image
def align_person(image, mask, train_images, train_masks):
    partial_image = np.zeros(MODEL_INPUT_SIZE)
    partial_fitted_mask = np.zeros(MASK_SIZE)

    partial_image[HEAD_Y:HEAD_Y + image.shape[0], HEAD_X:HEAD_X + MEAN_WIDTH] = image * mask
    partial_mask[HEAD_Y:HEAD_Y + image.shape[0], HEAD_X:HEAD_X + MEAN_WIDTH] = mask

    # get best base image
    masks_intersection = np.sum(train_masks[:, :HEAD_Y + image.shape[0]] * partial_mask, axis=0)
    masks_union = np.sum((train_masks[:, :HEAD_Y + image.shape[0]] + partial_mask) > 0, axis=0)
    best_fit = np.argmax(masks_intersection / masks_union)

    base_image = inpaint_image(train_images[best_fit], train_masks[best_fit], HEAD_Y + image.shape[0])

    return partial_image + (1 - partial_mask) * base_image    


def process_image(image, train_images, train_masks):
    # extract person mask using pspnet
    mask = get_image_mask(image)

    # get only area around person
    person_image, person_mask = forward_person_transform(image, mask)

    # prepare image to put into generative model
    person_image_aligned, person_mask_aligned = align_person(person_image, person_mask, train_images, train_masks)

    # get prediction from unboxing model
    prediction = get_person_prediction(person_image_aligned)

    # put prediction on old background and blend
    resulted_image = inverse_person_transform(image, mask, prediction, person_mask_aligned)

    return resulted_image

    