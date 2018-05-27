import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
import sys
import yaml
import glob

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, x_start_stop, y_start_stop, scale, cell_step, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, emit_all_windows=False):
    img_boxes = []
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    img_tosearch = img[y_start_stop[0]:y_start_stop[1],x_start_stop[0]:x_start_stop[1],:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = cell_step  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or emit_all_windows == True:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                img_boxes.append(((xbox_left+x_start_stop[0],ytop_draw+y_start_stop[0]),(xbox_left+x_start_stop[0]+win_draw,ytop_draw+y_start_stop[0]+win_draw)))
                
    return img_boxes
    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: hog_subsample.py image_search_path classifier_pickle.p params.yml")
        sys.exit(1)    
    ## Search for vehicles in images

    # load a pe-trained svc model from a serialized (pickle) file
    clf_pickle = pickle.load( open(sys.argv[2], "rb" ) )

    # get attributes of our svc object
    svc = clf_pickle["clf"]
    X_scaler = clf_pickle["scaler"]

    searchpath = sys.argv[1]
    example_images = glob.glob(searchpath)
    
    with open(sys.argv[3]) as f:
        params = yaml.load(f)
    
    feature_params = params['feature_params']
    tracker_params = params['tracker_params']
        
    img = mpimg.imread(example_images[0])
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    img_boxes = find_cars(img, tracker_params['regions'][0][0], tracker_params['regions'][0][1], tracker_params['scales'][0], tracker_params['cells_per_step'], svc, X_scaler, feature_params['orient'], feature_params['pix_per_cell'], feature_params['cell_per_block'], feature_params['spatial_size'], emit_all_windows=True)
    out_img = draw_boxes(draw_img, img_boxes)
    plt.imshow(out_img)
    #plt.savefig('./output_images/sliding_windows.jpg')
    plt.show()


    fig, ax = plt.subplots(int(len(example_images)/2),2)
    fig.set_size_inches(14,16)
    i = 0    
    for img_src in example_images:
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        img_boxes = find_cars(img, tracker_params['regions'][0][0], tracker_params['regions'][0][1], tracker_params['scales'][0], tracker_params['cells_per_step'], svc, X_scaler, feature_params['orient'], feature_params['pix_per_cell'], feature_params['cell_per_block'], feature_params['spatial_size'])
        out_img = draw_boxes(draw_img, img_boxes)
        ax[int(i/2)][int(i%2)].imshow(out_img)
        ax[int(i/2)][int(i%2)].set_title(img_src.split('\\')[-1])
        i+=1
    #plt.savefig('./output_images/sliding_window.jpg')
    plt.show()
