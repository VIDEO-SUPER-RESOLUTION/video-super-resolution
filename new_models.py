from __future__ import print_function, division

from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from advanced import HistoryCheckpoint, SubPixelUpscaling, non_local_block, TensorBoardBatch
import img_utils

import numpy as np
import os
import time
import warnings 

try:
    import cv2
    _cv2_available = True
except:
    warnings.warn('Could not load opencv properly. This may affect the quality of output images.')
    _cv2_available = False

train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)
def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_scale_type = "norm" # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128) -> Model:
        """
        Subclass dependent implementation.
        """
        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            if width is not None and height is not None:
                shape = (channels, width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier)
            else:
                shape = (channels, None, None)
        else:
            if width is not None and height is not None:
                shape = (width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier, channels)
            else:
                shape = (None, None, channels)

        init = Input(shape=shape)

        return init

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        """
        Standard method to train any of the models.
        """

        samples_per_epoch = img_utils.image_count()
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]
        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(img_utils.image_generator(train_path, scale_factor=self.scale_factor,
                                                           small_train_images=self.type_true_upscaling,
                                                           batch_size=batch_size),
                                 steps_per_epoch=samples_per_epoch // batch_size + 1,
                                 epochs=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                           scale_factor=self.scale_factor,
                                                                           small_train_images=self.type_true_upscaling,
                                                                           batch_size=batch_size),
                                 validation_steps=val_count // batch_size + 1)

        return self.model

    def evaluate(self, validation_dir):
        if self.type_requires_divisible_shape and not self.type_true_upscaling:
            _evaluate_denoise(self, validation_dir)
        else:
            _evaluate(self, validation_dir)

    def initial(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        import os
        from scipy.misc import imread, imresize, imsave
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

        img_dim_1, img_dim_2 = 0, 0

        
        # Use full image for super resolution
        img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
                                                                scale_factor)
        model = self.create_model(img_dim_2, img_dim_1, load_weights=True)
        if verbose: print("Model loaded.")
        return model




    def upscale(self, model, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        """
        Standard method to upscale an image.

        :param img_path:  path to the image
        :param save_intermediate: saves the intermediate upscaled image (bilinear upscale)
        :param return_image: returns a image of shape (height, width, channels).
        :param suffix: suffix of upscaled image
        :param patch_size: size of each patch grid
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        filename = "./final1/"+ path[0] + path[1]

        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

        img_dim_1, img_dim_2 = 0, 0

        
        # Use full image for super resolution
        img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
                                                                scale_factor)

        images = imresize(true_img, (img_dim_1, img_dim_2))
        images = np.expand_dims(images, axis=0)
        print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        # intermediate_img = None
        # if save_intermediate:
        #     if verbose: print("Saving intermediate image.")
        #     fn = path[0] + "_intermediate_" + path[1]
        #     intermediate_img = imresize(true_img, (init_dim_1 * scale_factor, init_dim_2 * scale_factor))
        #     imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.
        model = model
        # model = self.create_model(img_dim_2, img_dim_1, load_weights=True)
        # if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_dim_1 * scale_factor, init_dim_2 * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

    def __match_autoencoder_size(self, img_dim_1, img_dim_2, init_dim_1, init_dim_2, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_dim_2 * scale_factor) % 4 != 0) or ((init_dim_1 * scale_factor) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2 * scale_factor) // 4) * 4
                    img_dim_1 = ((init_dim_1 * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor
            else:
                # AE model and true upsampling
                if ((init_dim_2) % 4 != 0) or ((init_dim_1) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2) // 4) * 4
                    img_dim_1 = ((init_dim_1) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2, init_dim_1
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_dim_2, img_dim_1 = init_dim_2, init_dim_1
            else:
                # Not AE and not true upsampling
                img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor

        return img_dim_1, img_dim_2,
def _evaluate(sr_model : BaseSuperResolutionModel, validation_dir, scale_pred=False):
    """
    Evaluates the model on the Validation images
    """
    print("Validating %s model" % sr_model.model_name)
    if sr_model.model == None: sr_model.create_model(load_weights=True)
    if sr_model.evaluation_func is None:
        if sr_model.uses_learning_phase:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                  [sr_model.model.layers[-1].output])
        else:
            sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                              [sr_model.model.layers[-1].output])
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)
    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"
    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if sr_model.type_requires_divisible_shape:
                # Denoise models require precise width and height, divisible by 4

                if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                        or (width % 2 != 0) or (height % 2 != 0):
                    width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                    height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                    print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                          (sr_model.model_name, width, height))

                    y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32')
            x_width = width if not sr_model.type_true_upscaling else width // sr_model.scale_factor
            x_height = height if not sr_model.type_true_upscaling else height // sr_model.scale_factor

            x_temp = y.copy()

            if sr_model.type_scale_type == "tanh":
                x_temp = (x_temp - 127.5) / 127.5
                y = (y - 127.5) / 127.5
            else:
                x_temp /= 255.
                y /= 255.

            y = np.expand_dims(y, axis=0)

            img = img_utils.imresize(x_temp, (x_width, x_height),
                                     interp='bicubic')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (x_width, x_height), interp='bicubic')


            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            if scale_pred:
                if sr_model.type_scale_type == "tanh":
                    y_pred = (y_pred + 1) * 127.5
                else:
                    y_pred *= 255.

            if sr_model.type_scale_type == 'tanh':
                y = (y + 1) / 2

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))


def _evaluate_denoise(sr_model : BaseSuperResolutionModel, validation_dir, scale_pred=False):
    print("Validating %s model" % sr_model.model_name)
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    validation_path_set5 = validation_dir + "set5/"
    validation_path_set14 = validation_dir + "set14/"

    validation_dirs = [validation_path_set5, validation_path_set14]
    for val_dir in validation_dirs:
        image_fns = [name for name in os.listdir(val_dir)]
        nb_images = len(image_fns)
        print("Validating %d images from path %s" % (nb_images, val_dir))

        total_psnr = 0.0

        for impath in os.listdir(val_dir):
            t1 = time.time()

            # Input image
            y = img_utils.imread(val_dir + impath, mode='RGB')
            width, height, _ = y.shape

            if ((width // sr_model.scale_factor) % 4 != 0) or ((height // sr_model.scale_factor) % 4 != 0) \
                    or (width % 2 != 0) or (height % 2 != 0):
                width = ((width // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor
                height = ((height // sr_model.scale_factor) // 4) * 4 * sr_model.scale_factor

                print("Model %s require the image size to be divisible by 4. New image size = (%d, %d)" % \
                      (sr_model.model_name, width, height))

                y = img_utils.imresize(y, (width, height), interp='bicubic')

            y = y.astype('float32')
            y = np.expand_dims(y, axis=0)

            x_temp = y.copy()

            if sr_model.type_scale_type == "tanh":
                x_temp = (x_temp - 127.5) / 127.5
                y = (y - 127.5) / 127.5
            else:
                x_temp /= 255.
                y /= 255.

            img = img_utils.imresize(x_temp[0], (width // sr_model.scale_factor, height // sr_model.scale_factor),
                                     interp='bicubic', mode='RGB')

            if not sr_model.type_true_upscaling:
                img = img_utils.imresize(img, (width, height), interp='bicubic')

            x = np.expand_dims(img, axis=0)

            if K.image_dim_ordering() == "th":
                x = x.transpose((0, 3, 1, 2))
                y = y.transpose((0, 3, 1, 2))

            sr_model.model = sr_model.create_model(height, width, load_weights=True)

            if sr_model.evaluation_func is None:
                if sr_model.uses_learning_phase:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input, K.learning_phase()],
                                                          [sr_model.model.layers[-1].output])
                else:
                    sr_model.evaluation_func = K.function([sr_model.model.layers[0].input],
                                                          [sr_model.model.layers[-1].output])

            if sr_model.uses_learning_phase:
                y_pred = sr_model.evaluation_func([x, 0])[0][0]
            else:
                y_pred = sr_model.evaluation_func([x])[0][0]

            if scale_pred:
                if sr_model.type_scale_type == "tanh":
                    y_pred = (y_pred + 1) * 127.5
                else:
                    y_pred *= 255.

            if sr_model.type_scale_type == 'tanh':
                y = (y + 1) / 2

            psnr_val = psnr(y[0], np.clip(y_pred, 0, 255) / 255)
            total_psnr += psnr_val

            t2 = time.time()
            print("Validated image : %s, Time required : %0.2f, PSNR value : %0.4f" % (impath, t2 - t1, psnr_val))

            generated_path = predict_path + "%s_%s_generated.png" % (sr_model.model_name, os.path.splitext(impath)[0])

            if K.image_dim_ordering() == "th":
                y_pred = y_pred.transpose((1, 2, 0))

            y_pred = np.clip(y_pred, 0, 255).astype('uint8')
            img_utils.imsave(generated_path, y_pred)

        print("Average PRNS value of validation images = %00.4f \n" % (total_psnr / nb_images))
class ResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(ResNetSR, self).__init__("ResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 64
        self.mode = 2

        self.weight_path = "weights/ResNetSR %dX.h5" % (self.scale_factor)
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(ResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(64, (3, 3), activation='relu', padding='same', name='sr_res_conv1')(init)

        #x1 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv2')(x0)

        #x2 = Convolution2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2), name='sr_res_conv3')(x1)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = Add()([x, x0])

        x = self._upscale_block(x, 1)
        #x = Add()([x, x1])

        #x = self._upscale_block(x, 2)
        #x = Add()([x, x0])

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path, by_name=True)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(64, (3, 3), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = init._keras_shape[channel_dim]

        #x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
        #x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = UpSampling2D()(init)
        x = Convolution2D(self.n, (3, 3), activation="relu", padding='same', name='sr_res_filter1_%d' % id)(x)

        # x = Convolution2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
        #                            name='upsampling_deconv_%d' % id)(init)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="ResNetSR History.txt"):
        super(ResNetSR, self).fit(batch_size, nb_epochs, save_history, history_fn)
