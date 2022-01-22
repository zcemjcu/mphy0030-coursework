import numpy as np
from scipy import ndimage
from PIL import Image
import time

def distance_transform_np(data, voxel_dim=[2,0.5,0.5]):

    ''' Distance transform to calculate the euclidean distance
    between a 1 to its closest 0. 
    Input:
        data - a volumetric binary image
        voxel_dim - the dimensions of the binary image,
            here it is set to the size of the data given
            but can be changed
    Output:
        euc_transform - a 3D array containing the shortest
            euclidean distances given by: sqrt((x2-x1)^2 + (y2-y1)^2 
                                                + (d2-d1)^2)
            here the x, y, d correspond to the row, column and layer
            value respectively. The array is given as floats.            
    '''

    startTime = time.time() # to time the algorithm
    # creates an array of zeros as specified by voxel_dim
    euc_transform = np.zeros_like(data, dtype=float)
    # assigns variables for the sizes of the input data
    layers, height, width = data.shape[0], data.shape[1], data.shape[2]

    # the following loops are to shorten the input data where possible,
    # as the medical images can often have a lot of 0's (black) surrounding
    # the imaged portion, the outer 0's do not need to be considered, up to
    # the closest full layer of 0's before there are 1's

    # finds the lower layer of 0's that bounds the first layer with 1's
    for i in range(layers//2):
        data_layer = data[i]
        ones_in_layer = np.where(data_layer == 1)
        ones_in_layer = np.asarray(ones_in_layer).T

        if len(ones_in_layer) != 0:
            lower_layer_limit = i - 1
            break
    
    # finds the upper layer of 0's that bounds the last layer with 1's
    for i in range(layers-1, layers//2, -1):
        data_layer = data[i]
        ones_in_layer = np.where(data_layer == 1)
        ones_in_layer = np.asarray(ones_in_layer).T

        if len(ones_in_layer) != 0:
            upper_layer_limit = i + 1
            break

    # creates a new image without the unneccessary full 0 layers
    new_data = data[lower_layer_limit:upper_layer_limit+1]
    # obtains the locations of the 1's and 0's and converts them to arrays
    loc_of_ones = np.where(new_data == 1)
    ones = np.asarray(loc_of_ones).T # transposes the data
    loc_of_zeroes = np.where(new_data == 0)
    zeroes = np.asarray(loc_of_zeroes) # no transpose for vectorisation
    
    # to check that the loop is working
    count = len(ones)

    # arrays of the corresponding location of 0's, where d1 is the layer
    # number, x1 is the height number and y1 is the width number
    d1 = zeroes[0, :]
    x1 = zeroes[1, :]
    y1 = zeroes[2, :]

    for i in range(len(ones)):

        # takes each location of a one and retrieves its
        # layer, row and column value
        one_layer = ones[i]
        d2, x2, y2 = one_layer

        # assigns the shortest distance to infinity for 
        # comparison below
        shortest_dist = np.inf

        # calculates the distance between the selected location of a 1,
        # and all the locations of the 0's
        distance = ((x2*voxel_dim[1]-x1*voxel_dim[1])**2
                    + (y2*voxel_dim[2]-y1*voxel_dim[2])**2
                    + (d2*voxel_dim[0]-d1*voxel_dim[0])**2)

        # obtains the shortest distance from the distance array
        shortest_distance = distance.min()
        # square root for euclidean distance
        euc_dist = np.sqrt(shortest_distance)
        # assigns the shortest value found to its corresponding location
        euc_transform[d2+lower_layer_limit, x2, y2] = euc_dist

        print(count)
        count-=1

    executionTime = (time.time() - startTime) # outputs time taken for check
    print('Execution time in minutes: ' + str(executionTime/60))
    return euc_transform


def image_slice(data, slice_no, save=False, save_name=""):

    ''' Returns image of a slice from distance transformed data
    Inputs:
        data - the input image, array of values
        slice_no - the number of the slice the user wants
            saved
        save - Boolean, to decide whether to save the slice
            or not
        save_name - the name as which the slice should be 
            saved
    Outputs:
        im.show() - shows the user the slice
        im.save() - saves the slice as specified
    '''

    # retrieves the slice from the full image
    slice = data[slice_no]
    # outputs the distance transforms for specified slice
    print(slice)
    # scales the slice data to values between 0-255 for better display
    image = ((slice-np.amin(slice)) / (np.amax(slice)-np.amin(slice))) *255
    # assigns the values to int8 values as the float values use mode "F"
    # which is unsupported by a save to PNG file with pillow
    image = image.astype("int8")
    # retrieves the image
    im = Image.fromarray(image, mode="L")
    
    # saves and shows the file as specified
    if save:
        file_name = save_name + ".png"
        im.save(file_name)
        im.show()

    else:
        im.show()

    return 

# the time taken for my algorithm was approximately 2.5minutes on my desktop
# whereas the time taken for the ndimage algorithm was less than a second
# this is due to the ndimage function is compiled in a language such as C, and 
# because of this it can run the calculations at a much greater speed, compared
# to the vectorised method that I have applied
data = np.load("label_train00.npy") # loads segmentation file

np_output = distance_transform_np(data) # my edt algorithm

# scipy edt algorithm, sampling corresponds to 2mm slice distancing,
# and 0.5mm voxel distancing
edt_output = ndimage.distance_transform_edt(data, sampling=(2, 0.5, 0.5))

# saves the images for slices 8, 10, 14, 20 and 22 and saves them as shown
image_slice(edt_output, 8, save=True, save_name="ndimage_slice8"), image_slice(np_output, 8, save=True, save_name="myfunc_slice8")
image_slice(edt_output, 10, save=True, save_name="ndimage_slice10"), image_slice(np_output, 10, save=True, save_name="myfunc_slice10")
image_slice(edt_output, 14, save=True, save_name="ndimage_slice14"), image_slice(np_output, 14, save=True, save_name="myfunc_slice14")
image_slice(edt_output, 20, save=True, save_name="ndimage_slice20"), image_slice(np_output, 20, save=True, save_name="myfunc_slice20")
image_slice(edt_output, 22, save=True, save_name="ndimage_slice22"), image_slice(np_output, 22, save=True, save_name="myfunc_slice22")

# calculates the mean and standard deviation on the voxel level difference
voxel_diff = np.subtract(edt_output, np_output)
voxel_diff_mean = np.mean(voxel_diff)
voxel_diff_std = np.std(voxel_diff)
print("The voxel level difference mean is {0}, and the standard deviation is {1}.".format(voxel_diff_mean, voxel_diff_std))
