from skimage.measure import block_reduce
import cv2
import numpy as np

def iris_enhancement(img, illumination=False):  
    if illumination is True: # apply illumination adjustment
        #@the illumination estimation does not work well, so although I have coded them, I decided not to use them   
        # calculate the 4x32 mean_pool_transformed matrix
        mean_pool = block_reduce(img, (16,16),np.mean)

        # estimate the illumination by bicubic interpolation
        estimated_illumination = cv2.resize(mean_pool, (512, 64), interpolation =cv2.INTER_CUBIC)

        # subtract the estimated illumination from the original image. If we get negative value then set to 0
        enhanced_image = img - estimated_illumination
        enhanced_image = enhanced_image - np.amin(enhanced_image.ravel())  # rescale back to (0-255)
    
    elif illumination is False: # does not apply illumination adjustment
        enhanced_image = img - 0
        
    # perform the histogram equalization in each 32x32 region
    for row_index in range(0, enhanced_image.shape[0], 32):
        for col_index in range(0, enhanced_image.shape[1],32):
            sub_matrix = enhanced_image[row_index:row_index+32, col_index:col_index+32]
            # apply histogram equalization in each 32x32 sub block
            enhanced_image[row_index:row_index+32, col_index:col_index+32] = cv2.equalizeHist(sub_matrix.astype("uint8"))  
            
    return enhanced_image


