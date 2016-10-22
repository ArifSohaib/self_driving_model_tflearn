import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import scipy.misc
import h5py
import glob
import matplotlib.pyplot as plt

#frame size
nrows = 227
ncols = 227

frame_in =  tflearn.input_data(shape=[None, nrows, ncols,3])
net = conv_2d(frame_in, 24, 5, activation='elu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 36, 5, activation='elu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 48, 5, activation='elu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 3, activation='elu')
net = max_pool_2d(net, 2)
net = conv_2d(net, 64, 3, activation='elu')
net = max_pool_2d(net, 2)
shape = net.get_shape().as_list()
net = fully_connected(net, shape[1]*shape[2]*shape[3], activation='elu')
net = fully_connected(net, 1024, activation='elu')
net = fully_connected(net,1,activation='elu')
net = regression(net, optimizer='adam', loss='mean_square',metric='R2', learning_rate=0.001)
print(net)


def check_model(dfiles, network):
    model = tflearn.DNN(network,tensorboard_verbose=1, checkpoint_path='./model_nvidia/model.tfl.ckpt')
    for dfile in dfiles:
        with h5py.File(dfile) as h5f:
            #read the data from the file
            data = dict(h5f.items())
            #from the data dict, take the images and convert them to 1 byte(8-bits/uint8)
            images = np.array(data['images'].value, dtype=np.uint8)
            #change the image channels to RGB from BGR
            #second dimension has the color channels, which we reverse with ::-1 and take the rest of the dimensions as they are
            images = images[:,::-1,:,:].transpose((0,2,3,1))
            #take the targets and vehicle_states from he data
            targets = np.array(data['targets'].value)
            # print(targets[:,5].shape)
            vehicle_states = np.array(data['vehicle_states'].value)

            model.fit(images,targets[:,5].reshape(len(targets),1),validation_set=0.2,show_metric=True, snapshot_epoch=True,batch_size=20)
    predictions = model.predict(img_resized[100:110])
    print(predictions)
    print(targets[101:111,5])
    plt.plot(predictions,targets[101:111,5])
    plt.show()
dfiles = glob.glob('./data/*.h5')
check_model(dfiles, net)
