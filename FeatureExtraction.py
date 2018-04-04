from scipy import signal
import numpy as np

def defined_filter(size, sigma_x, sigma_y):

    f = 1.0/sigma_y   
    filter_mat = np.zeros((size,size))
    
    for xi in range(size):
        for yi in range(size):
            x = xi-size//2
            y = yi-size//2
            gaussian_value = 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-1.0/2 * (x**2/sigma_x**2 + y**2/sigma_y**2))
            M1 = np.cos(2*np.pi*f*np.sqrt(x**2+y**2))
            filter_mat[yi][xi] = gaussian_value * M1

    return filter_mat   

def execute_feature_extraction(input):
    
    ## function to extract the feature of 8x8 blocks
    def feature_extraction(filtered_img_1, filtered_img_2):
        V = []
        for row_index in range(0, filtered_img_1.shape[0], 8):  # use stride = 8
                for col_index in range(0, filtered_img_1.shape[1],8):  # use stride = 8
                    # process the first filtered image
                    sub_region_vec1 = abs(filtered_img_1[row_index:row_index+8, col_index:col_index+8].ravel())
                    # calculate m and sigma as denoeted in the paper
                    m1 = sub_region_vec1.mean()
                    sigma1 = 1/64 * (abs(sub_region_vec1-m1).sum())
                    V.append(m1)
                    V.append(sigma1)

                    # process the second filter image
                    sub_region_vec2 = abs(filtered_img_2[row_index:row_index+8, col_index:col_index+8].ravel())
                    # calculate m and sigma as denoeted in the paper
                    m2 = sub_region_vec2.mean() 
                    sigma2 = 1/64 * (abs(sub_region_vec2-m2).sum())
                    V.append(m2)
                    V.append(sigma2)
        return V  
    
    output = []
    filter1 = defined_filter(size=3, sigma_x=3, sigma_y=1.5)    # 3x3 defined gabor filter
    filter2 = defined_filter(size=3, sigma_x=4.5, sigma_y=1.5)  # 3x3 defined gabor filter
    
    # process the input data
    for img_enhanced in input:
        # get region of interest
        img = img_enhanced[:48,:]  
        
        # convolve the image by the two filters
        filtered_img_1 = signal.convolve2d(img, filter1, mode='same', boundary='wrap')
        filtered_img_2 =  signal.convolve2d(img, filter2, mode='same', boundary='wrap')
        
        # get the 1536x1 feature vector
        V = feature_extraction(filtered_img_1, filtered_img_2)
        
        #append the output
        output.append(V)
    
    return output