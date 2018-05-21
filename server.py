from PIL import Image
from pspnet import *
from model import cyclegan

import flask
import io

from align import *

app = flask.Flask(__name__)

pspnet = None
cyclegan_model = None


args = {'batch_size':1, 'fine_size':256, 'input_nc':3, 'output_nc':3, \
    'L1_lambda':10., 'dataset_dir':'tits2tits', 'use_resnet':True, \
    'use_lsga':True, 'max_size':50, 'ndf':64, 'ngf':64, 'phase':'test',
    'checkpoint_dir':'./checkpoint/'}

import tensorflow as tf

def load_psp():
    global pspnet

    pspnet = PSPNet101(nb_classes=21, input_shape=(473, 473),
                                       weights='pspnet101_voc2012')

def load_cg():
    global cyclegan_model

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)

    cyclegan_model = cyclegan(sess, args)
    cyclegan_model.init_load(args)


def predict_cg(image):
    global cyclegan_model

    out_var, in_var = (cyclegan_model.testB, cyclegan_model.test_A)
    fake_img = cyclegan_model.sess.run(out_var, feed_dict={in_var: image})

    return fake_img[0]


def process_psp(image):
    global pspnet

#    cimg = misc.imresize(image, (500, 500))

    cimg = image
    probs = pspnet.predict(cimg, True)

    cm = np.argmax(probs, axis=2)

    return 1 * (cm == 15)

train_images = None
train_masks = None

import uuid

def minverse_transform(images):
    return np.clip((images + 1.) * 127.5, 0, 255).astype(np.uint8)

def process_all(image):
    global pspnet

    global cyclegan_model

    print("start data processing")

#    fy = min(1, 400 / float(image.shape[0]))
#    print("resize image") 
#    image = cv2.resize(image, None, fx=fy, fy=fy)

    image_name = 'loaded_images/' + str(uuid.uuid4()) + '.jpg'
    cv2.imwrite(image_name, image)

    mask = process_psp(image)
    mask = np.array(mask > 0, np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("psp processed")
    print("psp results:", mask.shape, mask.sum())

    if mask.sum() < 20:
        return None

    cv2.imwrite('mask.png', mask * 255)

    print("forward transform")
    person, person_mask = forward_person_transform(image, mask)
    print("forward transform end")

    print("forward transform", person.shape)

#    person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

    print("alignment")
    aligned_image, aligned_mask = align_person(person, person_mask[..., None], train_images, train_masks)
    print("alignment end")

    mask_name = image_name[:-4] + '-mask' + '.jpg'
    cv2.imwrite(mask_name, aligned_image)

    print("start process cgan")
    cg_image = process_image(mask_name)
    print("end process cgan prepare")
#cg_image = cv2.resize(cg_image, (aligned_image.shape[0], aligned_image.shape[1]))
    print('images shapes', image.shape, mask.shape, cg_image.shape, aligned_mask.shape)
    predicted_pic = predict_cg(cg_image)
    print("cg prediction end", predicted_pic.shape, predicted_pic.mean(), predicted_pic.min(), predicted_pic.max())
    predicted_pic = minverse_transform(predicted_pic.copy())
    print("cg inverse transformed", predicted_pic.shape, predicted_pic.mean())
    print("cg alignes shape", aligned_mask.shape)
    predicted_pic = cv2.resize(predicted_pic, (aligned_mask.shape[1], aligned_mask.shape[0]))
    print("cg predict end", predicted_pic.shape, predicted_pic.mean())


    predicted_pic = cv2.cvtColor(predicted_pic, cv2.COLOR_RGB2BGR)
    cv2.imwrite('pred.png', predicted_pic)

    cg_name = image_name[:-4] + '-cg' + '.jpg'
    cv2.imwrite(cg_name, predicted_pic)

    print("start inverse transforn")
    print("inverse image", image.shape, image.dtype, image.max())
    print("inverse mask", mask.shape, mask.dtype, mask.max())
    print("inverse predicted pic",predicted_pic.shape,predicted_pic.dtype, predicted_pic.max())
    print("inverse aligned mask", aligned_mask.shape, aligned_mask.dtype, aligned_mask.max())
    result = inverse_person_transform(np.array(image), np.array(mask), np.array(predicted_pic), np.array(aligned_mask[..., 0], np.uint8))
    print("inverse end")

    fy = max(1, 400 / result.shape[0])
    result = cv2.resize(result, None, fx=fy, fy=fy)

    pred_name = image_name[:-4] + '_prediction.png'
    cv2.imwrite(pred_name, result)

    return pred_name, mask_name, cg_name

@app.route("/")
def test():
    return "test"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            img = np.array(image)
            try:            
                result = process_all(img)
                print('processed', result)
#print(img.shape)
            #image = process_psp(image)
            #print(image)

            #some process


                data["result_path"] = result[0]
                data['mask_path'] = result[1]
                data['cg_path'] = result[2]
                data["success"] = True
            except:
                pass

    return flask.jsonify(data)

from utilsc import load_test_data
import numpy as npo

def process_image(sample_file):
    sample_image = [load_test_data(sample_file, 256)]
    print("end process loading")
    sample_image = np.array(sample_image).astype(np.float32)

    return sample_image

if __name__ == "__main__":
    print("load and start")
    load_cg()
    img = process_image('img_test.jpg')
    print(img)
    predict_cg(img)

    load_psp()
    image = cv2.imread('img_test.jpg')
    pspnet.predict(image, True)

    train_images, train_masks = load_train_data('tits2tits_masked/trainA/')
    print("train_loaded", train_images.shape, train_masks.shape, train_masks.mean())

    app.run(host="0.0.0.0", port=80)
