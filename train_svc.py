import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import yaml
from lesson_functions import *
import cv2
import sys
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# NOTE: the last import is only valid for scikit-learn version >= 0.18
# for scikit-learn <= 0.18 use:
# from sklearn.cross_validation import train_test_split

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    img = mpimg.imread(cars[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict
    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: train_svc.py vehicle_image_folder_path not_vehicle_image_folder_path params.yml [C]\n  note: remember to use trailing '/' in folder path. e.g. vehicles/\n        if C is not defined GridSearchCV is executed and clasifier will not be saved")
        sys.exit(1)
    ## Step 1: make a list of images to read in
    # images are divided up into vehicles and non-vehicles folders (each of wich contains subfolder)
    # First locate vehicle images
    basedir = sys.argv[1]
    # Different folders represent different sources for images e.g. GTI, Kitti, generated from video
    image_types = os.listdir(basedir)
    cars = []
    for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))

    # Do the same thing for non-vehicle images
    basedir = sys.argv[2]
    image_types = os.listdir(basedir)
    notcars = []
    for imtype in image_types:
        notcars.extend(glob.glob(basedir+imtype+'/*'))
   
    data_info = data_look(cars, notcars)

    print('Data set includes a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')

    # Just for fun choose random car / not-car indices and plot example images   
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))
    
    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])


    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    #plt.savefig('./output_images/car_not_car.jpg')
    plt.show()

    
    params = {}

    ## Train a classifier
    
    with open(sys.argv[3]) as f:
        params = yaml.load(f)

    # Define feature parameters
    feature_params = params['feature_params']

    car_image_convert = convert_color(car_image, conv='RGB2'+feature_params['color_space'])
    notcar_image_convert = convert_color(notcar_image, conv='RGB2'+feature_params['color_space'])
    car_spatial_image = cv2.resize(car_image_convert, feature_params['spatial_size'])
    notcar_spatial_image = cv2.resize(notcar_image_convert, feature_params['spatial_size'])

    fig = plt.figure(figsize=(16,10))
    for i in range(3):
        plt.subplot(3,6,i*6+1)
        draw_img = np.copy(car_image)
        draw_img[:,:,:] = 0.5
        draw_img[:,:,i] = car_image_convert[:,:,i]
        draw_img = convert_color(draw_img, feature_params['color_space']+'2RGB')
        plt.imshow(draw_img)
        plt.title('Car Image CH-'+str(i+1))
        car_hog_features, car_hog_image =  get_hog_features(car_image_convert[:,:,i], feature_params['orient'], feature_params['pix_per_cell'], feature_params['cell_per_block'], vis=True)
        plt.subplot(3,6,i*6+2)
        plt.imshow(car_hog_image, cmap='hot')
        plt.title('Car Image CH-'+str(i+1)+' HOG')
        plt.subplot(3,6,i*6+3)
        plt.imshow(car_spatial_image[:,:,i], cmap='gray')
        plt.title('Car Image CH-'+str(i+1)+' Features')
        plt.subplot(3,6,i*6+4)
        draw_img = np.copy(notcar_image)
        draw_img[:,:,:] = 0.5
        draw_img[:,:,i] = notcar_image_convert[:,:,i]
        draw_img = convert_color(draw_img, feature_params['color_space']+'2RGB')
        plt.imshow(draw_img)
        plt.title('Not Car Image CH-'+str(i+1))
        notcar_hog_features, notcar_hog_image =  get_hog_features(notcar_image_convert[:,:,i], feature_params['orient'], feature_params['pix_per_cell'], feature_params['cell_per_block'], vis=True)
        plt.subplot(3,6,i*6+5)
        plt.imshow(notcar_hog_image, cmap='hot')
        plt.title('Car Image CH-'+str(i+1)+' HOG')
        plt.subplot(3,6,i*6+6)
        plt.imshow(notcar_spatial_image[:,:,i], cmap='gray')
        plt.title('Not Car Image CH-'+str(i+1)+' Features')
    fig.tight_layout()
    plt.show()
    #plt.savefig('./output_images/HOG_example.jpg')
    
    if len(sys.argv) < 5:
        t=time.time()
        n_samples = 2000

        random_idxs = np.random.randint(0,len(cars), n_samples)
        test_cars = np.array(cars)[random_idxs]
        test_notcars = np.array(notcars)[random_idxs]

        car_features = extract_features(test_cars, color_space=feature_params['color_space'], spatial_size=feature_params['spatial_size'],
                                        hist_bins=feature_params['hist_bins'], orient=feature_params['orient'],
                                        pix_per_cell=feature_params['pix_per_cell'], cell_per_block=feature_params['cell_per_block'], hog_channel=feature_params['hog_channel'],
                                        spatial_feat=feature_params['spatial_feat'], hist_feat=feature_params['hist_feat'], hog_feat=feature_params['hog_feat'])

        notcar_features = extract_features(test_notcars, color_space=feature_params['color_space'], spatial_size=feature_params['spatial_size'],
                                           hist_bins=feature_params['hist_bins'], orient=feature_params['orient'],
                                           pix_per_cell=feature_params['pix_per_cell'], cell_per_block=feature_params['cell_per_block'], hog_channel=feature_params['hog_channel'],
                                           spatial_feat=feature_params['spatial_feat'], hist_feat=feature_params['hist_feat'], hog_feat=feature_params['hog_feat'])
          
        print(time.time()-t,'Seconds to compute features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler only on the training data
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
    
        print('Using spatial binning of:', feature_params['spatial_size'])
        print('Using:',feature_params['orient'],'orientations',feature_params['pix_per_cell'],
              'pixels per cell and', feature_params['cell_per_block'],'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        parameters = {'C':[0.1, 1, 10]}
        svr = LinearSVC()
        clf = grid_search.GridSearchCV(svr, parameters)
        clf.fit(X_train, y_train)
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
        print(clf.best_params_)
    else:
        t=time.time()

        test_cars = cars
        test_notcars = notcars

        car_features = extract_features(test_cars, color_space=feature_params['color_space'], spatial_size=feature_params['spatial_size'],
                                        hist_bins=feature_params['hist_bins'], orient=feature_params['orient'],
                                        pix_per_cell=feature_params['pix_per_cell'], cell_per_block=feature_params['cell_per_block'], hog_channel=feature_params['hog_channel'],
                                        spatial_feat=feature_params['spatial_feat'], hist_feat=feature_params['hist_feat'], hog_feat=feature_params['hog_feat'])

        notcar_features = extract_features(test_notcars, color_space=feature_params['color_space'], spatial_size=feature_params['spatial_size'],
                                           hist_bins=feature_params['hist_bins'], orient=feature_params['orient'],
                                           pix_per_cell=feature_params['pix_per_cell'], cell_per_block=feature_params['cell_per_block'], hog_channel=feature_params['hog_channel'],
                                           spatial_feat=feature_params['spatial_feat'], hist_feat=feature_params['hist_feat'], hog_feat=feature_params['hog_feat'])
          
        print(time.time()-t,'Seconds to compute features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler only on the training data
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X_train and X_test
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)
        
        print('Using spatial binning of:', feature_params['spatial_size'])
        print('Using:',feature_params['orient'],'orientations',feature_params['pix_per_cell'],
              'pixels per cell and', feature_params['cell_per_block'],'cells per block')
        print('Feature vector length:', len(X_train[0]))
    
        # Use a linear SVC
        svc = LinearSVC(C=float(sys.argv[4]))
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        print(round(time.time()-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        clf_params = {}
        clf_params['clf'] = svc
        clf_params['scaler'] = X_scaler
        pickle.dump( clf_params, open( "svm_clf_params.p", "wb" ) )