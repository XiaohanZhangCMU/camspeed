import os, sys, time, argparse
import cv2
import h5py
import random
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pylab as plt
from skimage.transform import resize
import keras.utils.vis_utils as vutil
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import MaxPooling2D, UpSampling2D, Conv2D, Conv2DTranspose, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Lambda, add, LSTM, TimeDistributed, concatenate
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.applications.vgg16 import preprocess_input
# from utils import *

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

FRAME_H, FRAME_W = 112, 112
TIMESTEPS = 16

data_folder = '/scratch/users/xzhang11/aws/datasets/commaAI/data/'

sometime = lambda aug: iaa.Sometimes(0.3, aug)
sequence = iaa.Sequential([ #sometime(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 3.0
                            #sometime(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
                            #sometime(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 3.), per_channel=0.5)), # add gaussian noise to images
                            sometime(iaa.Dropout((0.0, 0.1))), # randomly remove up to 10% of the pixels
                            sometime(iaa.CoarseDropout((0.0, 0.1), size_percent=(0.01, 0.02), per_channel=0.2)),
                            #sometime(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
                          ],
                          random_order=True # do all of the above in random order
                             )


def normalize(image):
    return image - [104.00699, 116.66877, 122.67892]

def augment(image, flip, bright_factor):
    # random disturbances borrowed from IAA
    image = sequence.augment_image(image)
    
    # random brightness change
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    # random flip (vertical axis)
    if flip:
        image = cv2.flip(image, 1)
                
    return image


""" This is the class to generate batches for both training and validation.
"""
class BatchGenerator:
    def __init__(self, file_path, indices, batch_size, timesteps=1, shuffle=True, jitter = True, norm=True, overlap=False):
        self.file_path  = file_path
        self.batch_size = batch_size
        self.timesteps  = timesteps
        
        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.images = sorted(os.listdir(self.file_path ))
        self.labels = open(self.file_path + '../train.txt').readlines()
        
        self.indices = indices

    def get_gen(self):
        num_img = len(self.indices)
        
        l_bound = 0
        r_bound = self.batch_size if self.batch_size < num_img else num_img    
        
        if self.shuffle: np.random.shuffle(self.indices)

        while True:
            if l_bound == r_bound:
                l_bound = 0
                r_bound = self.batch_size if self.batch_size < num_img else num_img
                
                if self.shuffle: np.random.shuffle(self.indices)

            # the arrays which hold in the inputs and outputs
            x_batch = np.zeros((r_bound - l_bound, self.timesteps, FRAME_H, FRAME_W, 3))
            y_batch = np.zeros((r_bound - l_bound, 1))
            currt_inst = 0        

            for index in self.indices[l_bound:r_bound]:
                #if index > 2*self.timesteps:
                #    index -= np.random.randint(0, self.timesteps)
                
                # construct each input
                flip = (np.random.random() > 0.5)
                bright_factor = 0.5 + np.random.uniform() * 0.5
                
                for i in range(self.timesteps):
                    image = cv2.imread(self.file_path + self.images[index-self.timesteps+1+i])
                    heigh = image.shape[0]
                    image = image[np.concatenate([np.arange(heigh//3), np.arange(heigh*2//3,heigh)]),:,:]
                    image = cv2.resize(image.copy(), (FRAME_H, FRAME_W))
                    
                    if self.jitter: image = augment(image, flip, bright_factor)
                    if self.norm:   image = normalize(image)                    
                    x_batch[currt_inst, i] = image

                # construct each output
                speeds = [float(speed) for speed in self.labels[index-self.timesteps+1:index+1]]
                y_batch[currt_inst] = np.mean(speeds)

                currt_inst += 1
                
            yield x_batch, y_batch

            l_bound = r_bound
            r_bound = r_bound + self.batch_size
            if r_bound > num_img: r_bound = num_img
                
    def get_size(self):
        return len(self.indices)/self.batch_size 


""" CamSpeed Problem 
"""
class CamSpeed:
    """ Create training set and validation set from the given video data.
    """
    def prepdata(self, args):
        
        video_inp = data_folder + 'train.mp4'
        image = data_folder + '/frames/'
        if args.mode == "test":
            video_inp = data_folder + 'test.mp4'
            image = data_folder + '/test_frames/'
        video_reader = cv2.VideoCapture(video_inp)
        
        counter = 0
        while(True):
            ret, frame = video_reader.read()
            if ret == True:
                cv2.imwrite(image+str(counter).zfill(6)+'.png', frame)
                counter += 1
            else:
                break
            if not counter%100: print('processed %d images' % counter) 
        video_reader.release()

    def custom_loss(self, y_true, y_pred):
        loss = tf.squared_difference(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss
    
    def create_model(self):
        print("Construct Conv3d Architecture")
        self.model = Sequential()
        input_shape=(TIMESTEPS,FRAME_H,FRAME_W,3) # l, h, w, c
        
        # 1st layer group
        self.model.add(Conv3D(64, (3, 3, 3),  activation='relu', padding='same', name='conv1', input_shape=input_shape))
        self.model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
        
        # 2nd layer group
        self.model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))
        
        # 3rd layer group
        self.model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a'))
        self.model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
        
        # 4th layer group
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a'))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))
        
        # 5th layer group
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a'))
        self.model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b'))
        self.model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
        self.model.add(Flatten())
        
        # FC layers group
        self.model.add(Dense(4096, activation='relu', name='fc6'))
        self.model.add(Dropout(.5))
        #self.model.add(Dense(4096, activation='relu', name='fc7'))
        #self.model.add(Dropout(.5))
        self.model.add(Dense(1,    activation='linear', name='fc8'))
        
        self.model.summary()
    
    
    """ Load weights pretrained on the Sports-1M dataset on first a few layers
    """
    def load_pretrained_weights(self):
        sports_1m = h5py.File(data_folder+'c3d-sports1M_weights.h5', mode='r')
        
        print('model has size of %d' % len(self.model.layers))
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            layer_name = 'layer_' + str(i)
            print('layer name = %s' % layer.name)
            
            weights = sports_1m[layer_name].values()
            weights = [weight.value for weight in weights]
            weights = [weight if len(weight.shape) < 4 else weight.transpose(2,3,4,1,0) for weight in weights]
            
            layer.set_weights(weights)
            
            # ignore the last 2 layer, 1 dropout and 1 dense
            if i > len(self.model.layers) - 3:
                break

    def test(self, BATCH_SIZE = 4):

        self.model.load_weights(data_folder+'weight_c3d.h5')

        """
        indices = range(TIMESTEPS-1, len(os.listdir(data_folder + 'test_frames/')), TIMESTEPS)
        test_indices = list(indices[0:int(len(indices))])
        gen_test = BatchGenerator(data_folder+'test_frames/', test_indices, batch_size=BATCH_SIZE, timesteps=TIMESTEPS)
        print("I am here")
        preds = self.model.predict_generator(generator=gen_test.get_gen(), steps=len(indices), max_queue_size=10, workers=1, use_multiprocessing=True, verbose=1) 
        np.savetxt('test.txt', preds)
        """

        video_inp = data_folder + 'test.mp4'
        video_reader = cv2.VideoCapture(video_inp)
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        label_out = open('test.txt', 'w')
        h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        x_batch_original = np.zeros((TIMESTEPS, h, w, 3))
        x_batch = np.zeros((1, TIMESTEPS, FRAME_H, FRAME_W, 3))
        frame_counter = 0
        frame_counter_all = 0
        curr_speed = 0.
        acc_error = 0
        
        while(True):
            ret, image = video_reader.read()
        
            if ret == True:
                if frame_counter_all > -1: # only start processing from certain frame count
                    x_batch_original[frame_counter] = image
        
                    heigh = image.shape[0]
                    image = image[np.concatenate([np.arange(heigh//3), np.arange(heigh*2//3,heigh)]),:,:]
                    image = cv2.resize(image.copy(), (FRAME_H, FRAME_W))
                    image = normalize(image)
        
                    x_batch[0, frame_counter] = image
                    curr_speed = self.model.predict(x_batch)[0][0]
                    label_out.write(str(curr_speed) + '\n')
                    print('predict %d / %d test frames' % (frame_counter_all, total_frames))
            else:
                break
            
            frame_counter_all += 1
        
        video_reader.release()
        label_out.close()

    
    """ Train the network after loading pretrained weights
    """
    def train(self, split_ratio=0.9, EPOCHS=5, BATCH_SIZE=4):
        self.load_pretrained_weights()
        
        early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, mode='min', verbose=1)
        checkpoint  = ModelCheckpoint(data_folder+'weight_c3d.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
        
        indices = range(TIMESTEPS-1, len(os.listdir(data_folder + 'frames/')), TIMESTEPS)
        np.random.shuffle(list(indices))
        
        train_indices = list(indices[0:int(len(indices)*split_ratio)])
        valid_indices = list(indices[int(len(indices)*split_ratio):])
        
        gen_train = BatchGenerator(data_folder+'frames/', train_indices, batch_size=BATCH_SIZE, timesteps=TIMESTEPS)
        gen_valid = BatchGenerator(data_folder+'frames/', valid_indices, batch_size=BATCH_SIZE, timesteps=TIMESTEPS, jitter = False)
        
    
        tb_counter  = max([int(num) for num in os.listdir(data_folder+'logs/speed/')] or [0]) + 1
        tensorboard = TensorBoard(log_dir=data_folder+'logs/speed/' + str(tb_counter), histogram_freq=0, write_graph=True, write_images=False)
        
        sgd = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
        self.model.compile(loss=self.custom_loss, optimizer=sgd)
        
        #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #model.compile(loss=custom_loss, optimizer=adam)
        
        #rms = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
        #model.compile(loss=custom_loss, optimizer=rms)
        
        self.model.fit_generator(generator = gen_train.get_gen(),
                            steps_per_epoch = gen_train.get_size(), 
                            epochs  = EPOCHS, 
                            verbose = 1,
                            validation_data = gen_valid.get_gen(), 
                            validation_steps = gen_valid.get_size(), 
                            callbacks = [early_stop, checkpoint, tensorboard], 
                            max_q_size = 8)

    def main(self, args):
        if args.prepdata:
            self.prepdata(args)
            return

        # compile model
        self.create_model()

        if args.mode == "train":
            if args.resume: #load existing weights
                self.load_weights()
            #start training session
            self.train(split_ratio=args.split_ratio, EPOCHS=args.EPOCHS, BATCH_SIZE=args.BATCH_SIZE)
        
        #test the model: generate submission file
        elif args.mode == "test":
            self.test(BATCH_SIZE=args.BATCH_SIZE)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("video_file",
    #                     help="video file name")
    # parser.add_argument("speed_file",
    #                     help="speed data file name")
    parser.add_argument("--mode", choices=["train", "test"], default='train',
                        help="Train or Test model")
    parser.add_argument("--resume", action='store_true',
                        help="resumes training")
    parser.add_argument("--prepdata", action='store_true',
                        help="preprocess video data")

    parser.add_argument("--split_ratio", type=float, default=0.9,
                        help="percentage of train data for validation")
    parser.add_argument("--EPOCHS", type=int, default=5,
                        help="epochs")
    parser.add_argument("--BATCH_SIZE", type=int, default=4,
                        help="batch_size")

    args = parser.parse_args()

    camspeed = CamSpeed()
    camspeed.main(args)


