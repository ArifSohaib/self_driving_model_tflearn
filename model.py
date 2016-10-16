import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import scipy.misc
import h5py
import glob
import matplotlib.pyplot as plt

#generate fake images

#generate fake data

#image goes into conv model which extracts out the impoartant features eg signs, other vechicles or obsticles

#flatten this out

#append the other data to the flattened layer

#output some small series of final layers eg. values for brakes, acceleration and steering
#note that brakes and acceleration could be combined

#frame_size
nrows = 64
ncols = 64

#inputs are
real_in = tflearn.input_data(shape=[None,6])
#video frame in
frame_in =  tflearn.input_data(shape=[None, 3, nrows, ncols])

#convolution for frame input
conv1 = conv_2d(frame_in, 32, 3, activation='relu')
conv1 = max_pool_2d(conv1, 2)
#fully_connected layer in tflearn automatically flattens conv layer input
#size = (W - F + 2P)/S+1 = (64 - 3 + 2P)/2 + 1 =
# print(conv1.get_shape().as_list())
flatten = fully_connected(conv1,2*32*32)
merge_data = tflearn.layers.merge_ops.merge([flatten, real_in],mode='concat')
output =  fully_connected(merge_data,2,activation='linear')

network = regression(output, optimizer='adam', loss='mean_square', learning_rate=0.001)
# print(network)
# tflearn.DNN(network)
dfiles = glob.glob('./data/*.h5')
def check_model(dfiles, network,n_samples, n_rows, n_cols):
    # fake_real = np.random.random((n_samples, 6))
    # fake_frame = np.random.random((n_samples, 3, n_rows, n_cols))
    # fake_output = np.random.random((n_samples, 2))
    model = tflearn.DNN(network,tensorboard_verbose=1)
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
            img_resized = img_resized.transpose((0,3,1,2))
            #take the targets and vehicle_states from he data
            targets = np.array(data['targets'].value)
            vehicle_states = np.array(data['vehicle_states'].value)
            #for each target
            model.fit([vehicle_states,img_resized],targets[:,4:],validation_set=0.2,show_metric=True, snapshot_epoch=True,batch_size=128)
    predictions = model.predict([vehicle_states[1:10,:],img_resized[1:10]])
    print(predictions)
    print(targets[2:11])
    plt.imshow(predictions)
    plt.show()
    # print(model.get_weights())
check_model(dfiles, network, 1000, 64, 64)
