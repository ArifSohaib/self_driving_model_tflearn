import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
#generate fake images

#generate fake data

#image goes into conv model which extracts out the impoartant features eg signs, other vechicles or obsticles

#flatten this out

#append the other data to the flattened layer

#output some small series of final layers eg. values for brakes, acceleration and steering
#note that brakes and acceleration could be combined

#frame_size
nrows = 227
ncols = 227

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
def check_model(network,n_samples, n_rows, n_cols):
    fake_real = np.random.random((n_samples, 6))
    fake_frame = np.random.random((n_samples, 3, n_rows, n_cols))
    fake_output = np.random.random((n_samples, 2))
    model = tflearn.DNN(network)
    return model.fit([fake_real,fake_frame],fake_output)
check_model(network, 1000, 227, 227)
