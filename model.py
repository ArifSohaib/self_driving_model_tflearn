import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import scipy.misc
import h5py
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.animation as animation

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
    #fully_connected layer in tflearn automatically flattens conv layer input
    #size = (W - F + 2P)/S+1 = (64 - 3 + 2P)/2 + 1 =
    # print(conv1.get_shape().as_list())
    net = fully_connected(net, shape[1]*shape[2]*shape[3], activation='elu')
    # net = fully_connected(net,2*16*16, activation='elu')
    net = fully_connected(net, 1000, activation='elu')
    net = fully_connected(net, 256, activation='elu')
    net = fully_connected(net, 64, activation='elu')
    # merge_data = tflearn.layers.merge_ops.merge([flatten, real_in],mode='concat')
    net =  fully_connected(net,1,activation='tanh')

    net = regression(net, optimizer='adam', loss='mean_square', metric='R2', learning_rate=0.001)
    return net

def visualize_graph(targets, predictions):
    """
    Makes visualizations using targets and predictions
    """
    #convert targets and predictions to np array
    #dimension 0 is the number of files, since the array was reshaped earlier, dim 2 is always 1(1 prediction) and dimension 1 is number of predictions for each file
    predictions = np.array(predictions).transpose((0,2,1))
    #plot the predictions for each file
    for i in range(predictions.shape[0]):
        plt.figure()
        plt.plot(predictions[i],'o')
        plt.show()

    #reshape targets to the same shape as predictions for easy plotting
    targets = np.array(targets)
    targets = targets.reshape(targets.shape[0], 1, targets.shape[1])
    for i in range(targets.shape[0]):
        plt.figure()
        plt.plot(targets[i], 'o')
        plt.show()

    #plot the targets against the predictions
    for i in range(predictions.shape[0]):
        plt.figure()
        plt.plot(predictions[i].reshape(100),'x', targets[i].reshape(100),'o')
        plt.show()

def get_lines(targets, predictions):
    """
    shows target and predicted steering
    """
    #map predictions to lines
    predictions = np.array(predictions).transpose((0,2,1))
    targets = np.array(targets)
    targets = targets.reshape(targets.shape[0],1, targets.shape[1])
    lines_pred = []
    lines_target = []
    for i in range(predictions.shape[0]):
        lines_pred.append([list(map(get_point,p)) for p in predictions[i]])
        lines_target.append([list(map(get_point,p)) for p in targets[i]])
    return lines_pred, lines_target

def visualize_image(image, target_lines, predicted_lines):
    #this is the first image in the last file read
    im = Image.fromarray(image[1])
    draw = ImageDraw.Draw(im)
    #in predicted_lines[-1] gets the last file, and the last 0 gets the image/data number
    draw.line((32,63, predicted_lines[-1][0][1],predicted_lines[-1][0][1]), fill=12)
    draw.line((32,63, target_lines[-1][0][1],target_lines[-1][0][1]), fill=255)
    plt.imshow(im,interpolation='nearest')
    plt.show()

def visualize_animation(images, target_lines, predicted_lines):
    #draw an empty canvas for the images
    figure = plt.figure()
    #the canvas is painted with zeros in the given size
    imageplot = plt.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
    #function to generate images for the animation
    def next_frame(i):
        im = Image.fromarray(images[i])
        draw = ImageDraw.Draw(im)
        draw.line((32,63, target_lines[-1][i][0],target_lines[-1][i][1]),fill=(255,0,0,128))
        draw.line((32,63, predicted_lines[-1][i][0],predicted_lines[-1][i][1]),fill=(0,255,0,128))
        imageplot.set_array(im)
        return imageplot,
    animate = animation.FuncAnimation(figure, next_frame, frames=range(len(images)-1), interval=10, blit=False)
    plt.show()

def get_point(s,start=0,end=63,height= 16):
    """
    Map given point(prediction) to between 0 and 63
    Args:
        s: the prediction point
        start: the minimum possible value of the point
        end: the maximum possible value of the point
        height: the length of the line
    """
    X = int(s*(end-start))
    if X < start:
        X = start
    if X > end:
        X = end
    return (X,height)

if __name__ == '__main__':
    network = build_network()
    dfiles = glob.glob('./data/*.h5')
    model = fit_model(dfiles, network)
    predictions = []
    targets = []
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
            targets.append(data['targets'][:100,5])

    predicted_lines, target_lines = get_lines(targets, predictions)
