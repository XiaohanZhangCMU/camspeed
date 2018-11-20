import cv2 
import h5py
from imgaug import augmenters as iaa

def write_hdf5(arr, outfile):
	"""
	Write an numpy array to a file in HDF5 format.
	"""
	with h5py.File(outfile, "w", libver='latest') as f:
		f.create_dataset("image", data=arr, dtype=arr.dtype)

def load_hdf5(infile):
	"""
	Load a numpy array stored in HDF5 format into a numpy array.
	"""
	with h5py.File(infile, "r", libver='latest') as hf:
		return hf["image"][:]

# Visualization makes sure generator works. 
def test_gen(ts):
    indices = range(ts-1, len(os.listdir(data_folder + 'frames/')), ts)
    gen_train = BatchGenerator(data_folder,indices,batch_size=4,timesteps=ts)
    generator = gen_train.get_gen()
    show(generator.next()[0][0][0])

def show(image, cmap='gray', ax=None):
    if ax is None:
        plt.figure()
    plt.imshow(image[:,:,::-1].astype('uint8'), cmap=cmap)


