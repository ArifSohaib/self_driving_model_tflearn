import h5py
import matplotlib.pyplot as plt
import glob
import numpy as np

#load the data files using glob
dfiles = glob.glob('./data/*.h5')
#print file names for debugging if needed
#print(dfiles)
'''TODO: currently we are using only 1 file as the whole dataset is 80 GB. When the model is done, we need to download more of the dataset'''

for dfile in dfiles:
    h5f = h5py.File(dfile, 'r')
    data = dict(h5f.items())
#convert from float32 to uint8
    images = np.array(data['images'].value, dtype=np.uint8)
    targets = np.array(data['targets'].value)
    vehicle_states = np.array(data['vehicle_states'].value)
    #clear the data to save memory
    # data = None

'''TODO: write somthing to read data in batches'''

def main():
    print(data['images'])
    print(data['images'].value[10].shape)
    print(data.keys())
    plt.imshow(data['images'].value[np.random.choice(range(1000))].transpose((1,2,0))[:,:,::-1])
    plt.show()
    print(data['targets'].shape)
    plt.figure()
    plt.imshow(images[np.random.choice(range(1000))].transpose((1,2,0))[:,:,::-1])
    plt.show()
    print(vehicle_states[11])
    print(targets[11])

if __name__ == '__main__':
    main()
