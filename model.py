import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import scipy.misc
import h5py
import glob
import matplotlib.pyplot as plt

#image goes into conv model which extracts out the impoartant features eg signs, other vechicles or obsticles

#flatten this out

#append the other data to the flattened layer

#output some small series of final layers eg. values for brakes, acceleration and steering
#note that brakes and acceleration could be combined

def fit_model(dfiles, network):
    # fake_real = np.random.random((n_samples, 6))
    # fake_frame = np.random.random((n_samples, 3, n_rows, n_cols))
    # fake_output = np.random.random((n_samples, 2))
    model = tflearn.DNN(network,tensorboard_verbose=1, checkpoint_path='./model/model.tfl.ckpt')
    for dfile in dfiles:
        with h5py.File(dfile) as h5f:
            #read the data from the file
            data = dict(h5f.items())
            #from the data dict, take the images and convert them to 1 byte(8-bits/uint8)
            images = np.array(data['images'].value, dtype=np.uint8)
            #change the image channels to RGB from BGR
            #second dimension has the color channels, which we reverse with ::-1 and take the rest of the dimensions as they are
            images = images[:,::-1,:,:]
            #create an array to store resized images
            img_resized = np.zeros((len(images),64,64, 3),dtype=np.uint8)
            for idx, img in enumerate(images):
                img_resized[idx] = scipy.misc.imresize(img, (64,64), 'cubic', 'RGB')
            images = None
            # img_resized = img_resized.transpose((0,3,1,2))
            #take the targets and vehicle_states from he data
            targets = np.array(data['targets'].value)
            # print(targets[:,5].shape)
            vehicle_states = np.array(data['vehicle_states'].value)

            model.fit(img_resized,targets[:,5].reshape(len(targets),1),validation_set=0.2,show_metric=True, snapshot_epoch=True,batch_size=100, snapshot_step=500)

    return model

def get_predictions(model, input_images):
    predictions = model.predict(input_images)
    return predictions

def build_network():
    #frame_size
    nrows = 64
    ncols = 64

    #inputs are
    # real_in = tflearn.input_data(shape=[None,4])
    #video frame in
    frame_in =  tflearn.input_data(shape=[None, nrows, ncols,3])

    #convolution for frame input
    net = conv_2d(frame_in, 32, 3, activation='elu')
    net = max_pool_2d(net, 2)
    net = conv_2d(frame_in, 16, 5, activation='elu')
    net = max_pool_2d(net, 2)
    #fully_connected layer in tflearn automatically flattens conv layer input
    #size = (W - F + 2P)/S+1 = (64 - 3 + 2P)/2 + 1 =
    # print(conv1.get_shape().as_list())
    net = fully_connected(net,2*16*16, activation='elu')
    net = fully_connected(net, 1000, activation='elu')
    net = fully_connected(net, 256, activation='elu')
    net = fully_connected(net, 64, activation='elu')
    # merge_data = tflearn.layers.merge_ops.merge([flatten, real_in],mode='concat')
    net =  fully_connected(net,1,activation='linear')

    net = regression(net, optimizer='adam', loss='mean_square', metric='R2', learning_rate=0.001)
    return net

def main():
    network = build_network()
    dfiles = glob.glob('./data/*.h5')
    model = fit_model(dfiles, network)
    predictions = []
    for dfile in dfiles:
        with h5py.File(dfile) as h5f:
            data = dict(h5f.items())
            images = np.array(data['images'].value, dtype=np.uint8)
            images = images[:,::-1,:,:]
            #create an array to store resized images
            img_resized = np.zeros((len(images),64,64, 3),dtype=np.uint8)
            for idx, img in enumerate(images):
                img_resized[idx] = scipy.misc.imresize(img, (64,64), 'cubic', 'RGB')
            images = None

            predictions.append(get_predictions(model, img_resized[:100]))

if __name__ == '__main__':
    main()
