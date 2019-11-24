import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model,load_model
from keras.layers import Conv2D, Input, add
from keras.optimizers import Adam, sgd
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy
import matplotlib.image as mpimg
import cv2
import h5py


# ==============================================================================
# init params
# ==============================================================================
def dssimloss(y_true, y_pred):
    ssim2 = tf.image.ssim(y_true, y_pred, 1.0)
    return K.mean(1 - ssim2)

IMG_SIZE = (None, None, 1)
SCALE_FACTOR = 3
CHECKPOINT_PATH = "./checkpoints/weights-improvement-{epoch:02d}-{PSNR:.2f}.hdf5"
GT = 'GT_image'
GT_NAME = GT + ".png"
ILR_NAME = GT + "ILR.jpg"
OUTPUT_NAME = GT + "predicted.jpg"
LOSS = dssimloss
LOSS_NAME = 'dssimloss'
EPOCHS = 5000
DATA_TRAIN = "./train-s3-91.h5"
DATA_TEST = "./test-s3.h5"
filters = {'f1': 9, 'f2': 5, 'f3': 5}
filters_num = {'n1': 128, 'n2': 64, 'n3': 1}
MODEL_NAME = 'model-%s' % LOSS_NAME + '-%d' % EPOCHS + 'epoch-%d' % filters['f1'] + '-%d' % filters['f2'] + '-%d' % \
             filters['f3'] + 'scale%d' % SCALE_FACTOR + 'final.pth'
BATCH_SIZE = 128
TRAIN = False


# ==============================================================================
# define functions
# ==============================================================================
def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


def SSIM(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, 1.0)


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label


# ==============================================================================
# define the SRCNN model
# ==============================================================================
def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(filters=64, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=32, kernel_size=(1, 1), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))

    # define optimizer
    adam = Adam(lr=0.0003)
    # compile model
    SRCNN.compile(optimizer=adam, loss=LOSS, metrics=[PSNR, SSIM])
    return SRCNN


model = model()

checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor=PSNR, verbose=1, mode='max')
callbacks_list = [checkpoint]

model.summary()
# ==============================================================================
# train modelgn
# ==============================================================================
ILR_train, HR_train = read_training_data(DATA_TRAIN)
ILR_test, HR_test = read_training_data(DATA_TEST)
# import keras.preprocessing.image.ImageDataGenerator
# datagen = ImageDataGenerator(
#    featurewise_center=True,
#    rotation_range=[90,180,270],
#    horizontal_flip=True)
if TRAIN:
    network_history = model.fit(ILR_train, HR_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(ILR_test, HR_test))
    model.save(MODEL_NAME)

if not TRAIN:
    def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


    def SSIM(y_true, y_pred):
        return tf.image.ssim(y_true, y_pred, 1.0)


    model = load_model(MODEL_NAME, custom_objects={'dssimloss': dssimloss, 'PSNR': PSNR, 'SSIM': SSIM})
    # history = model.history

# ==============================================================================
# plot results
# ==============================================================================
# losses = history['loss']
# plt.plot(losses)
# plt.xlabel('epoch')
# plt.ylabel('loss')
#
# psnr = history['PSNR']
# plt.figure()
# plt.plot(psnr)
# plt.xlabel('epoch')
# plt.ylabel('psnr')
#
# acc = history['SSIM']
# plt.figure()
# plt.plot(psnr)
# plt.xlabel('epoch')
# plt.ylabel('SSIM')
#
# # validation ____________________________________
# plt.figure()
# val_loss = history['val_loss']
# plt.plot(val_loss)
# plt.xlabel('epoch')
# plt.ylabel('val_loss')
#
# val_psnr = history['val_PSNR']
# plt.figure()
# plt.plot(val_psnr)
# plt.xlabel('epoch')
# plt.ylabel('val_PSNR')
#
# val_acc = history['val_SSIM']
# plt.figure()
# plt.plot(psnr)
# plt.xlabel('epoch')
# plt.ylabel('val_SSIM')
# print("Done training!!!")

# ==============================================================================
# predict
# ==============================================================================
srcnn_model = model
# srcnn_model.load_weights("3051crop_weight_200.h5")
GT_image = cv2.imread(GT_NAME, cv2.IMREAD_COLOR)
GT_shape = GT_image.shape
GT_image = cv2.imread(GT_NAME, cv2.IMREAD_COLOR)
img = cv2.cvtColor(GT_image, cv2.COLOR_BGR2YCrCb)
shape = img.shape
Y_img = cv2.resize(img[:, :, 0], (int(shape[1] / SCALE_FACTOR), int(shape[0] / SCALE_FACTOR)), cv2.INTER_CUBIC)
Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
img[:, :, 0] = Y_img
img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(ILR_NAME, img)

Y = numpy.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
Y[0, :, :, 0] = Y_img.astype(float) / 255.
pre = srcnn_model.predict(Y, batch_size=1) * 255.
pre[pre[:] > 255] = 255
pre[pre[:] < 0] = 0
pre = pre.astype(numpy.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img[:, :, 0] = pre[0, :, :, 0]
img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(OUTPUT_NAME, img)



#_----------------------------------------------


# ===========================================================================
# psnr calculation
# ============================================================================

im1 = cv2.imread(GT_NAME, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
im2 = cv2.imread(ILR_NAME, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[:, :, 0]


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    img_arr1 = numpy.array(im1).astype('float32')
    img_arr2 = numpy.array(im2).astype('float32')
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    psnr = tf.constant(255 ** 2, dtype=tf.float32) / mse
    result = tf.constant(10, dtype=tf.float32) * log10(psnr)
    with tf.Session():
        result = result.eval()
    return result


p_ILR = psnr(im1, im2)
p_SR = psnr(im1, im3)
print('the psnr for ILR is : ', p_ILR)
print('the psnr for SR is : ', p_SR)
# ===========================================================================
# save and plot
# ============================================================================
dim = (1, 3)
figsize = (15, 5)
plt.figure(figsize=figsize)

plt.subplot(dim[0], dim[1], 1)
GT = mpimg.imread(GT_NAME)
plt.imshow(GT)
plt.title('Ground truth' + '-scale : %d' % SCALE_FACTOR)
plt.axis('off')

plt.subplot(dim[0], dim[1], 2)
ILR = mpimg.imread(ILR_NAME)
plt.title('Interpolation' + '-PSNR : %s dB' % p_ILR)
plt.imshow(ILR)
plt.axis('off')

plt.subplot(dim[0], dim[1], 3)
SR = mpimg.imread(OUTPUT_NAME)
plt.imshow(SR)
plt.title('Super resolution' + '-PSNR : %s dB' % p_SR)
plt.axis('off')

plt.tight_layout()
plt.savefig('%s-compare' % GT_NAME + '-%d.png' % SCALE_FACTOR)

plt.show()
from skimage.measure import compare_ssim

print(compare_ssim(im1, im3))
