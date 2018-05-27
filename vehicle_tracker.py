from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import sys
import yaml
import pickle
import numpy as np
import cv2
from hog_subsample import find_cars

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class VehicleTracker():
     # when starting a new instance please be sure to specify all unassigned variables
    def __init__(self, search_regions, search_scales, cells_per_step, clf, scaler, feature_params, smooth_factor=3, threshold=1, key_frame_interval=1):
        self.regions = search_regions
        self.scales = search_scales
        self.cell_step = cells_per_step
        self.svc = clf
        self.X_scaler = scaler
        self.feature_params = feature_params
        self.smooth_factor = smooth_factor
        self.threshold = threshold
        self.recent_windows = []
        self.integrated_heatmap = None
        self.labels = None
        self.frame_interval = key_frame_interval
        self.frame = 0
        
    def find_vehicles(self, img):
        orient = self.feature_params['orient']
        pix_per_cell = self.feature_params['pix_per_cell']
        cell_per_block = self.feature_params['cell_per_block']
        spatial_size = self.feature_params['spatial_size']
        heatmap = np.zeros_like(img[:,:,0])
        if self.frame%self.frame_interval > 0 and self.labels[1] >= 1:
            # Find pixels with label values
            nonzero = (self.labels[0] > 0).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            min_x = max(0,np.min(nonzerox)-32)
            max_x = min(img.shape[1],np.max(nonzerox)+32)-1
            min_y = max(0,np.min(nonzeroy)-32)
            max_y = min(img.shape[0],np.max(nonzeroy)+32)-1
            for scale in self.scales:
                img_boxes = find_cars(img, [min_x,max_x], [min_y,max_y], scale, self.cell_step, self.svc, self.X_scaler, orient, pix_per_cell, cell_per_block, spatial_size) 
                self.recent_windows.append(img_boxes) 
        else:
            for search_region, scale in zip(self.regions,self.scales):
                img_boxes = find_cars(img, search_region[0], search_region[1], scale, self.cell_step, self.svc, self.X_scaler, orient, pix_per_cell, cell_per_block, spatial_size) 
                self.recent_windows.append(img_boxes)        
        for i in range(min(len(self.recent_windows),self.smooth_factor*len(self.scales))):
            add_heat(heatmap,self.recent_windows[-i])
        if len(self.recent_windows) > self.smooth_factor*2*len(self.scales):
            self.recent_windows = self.recent_windows[-self.smooth_factor*2*len(self.scales):] # clip off to reduce memory
        apply_threshold(heatmap,min(len(self.recent_windows),self.threshold))
        self.integrated_heatmap = heatmap
        self.labels = label(heatmap)
        self.frame += 1
        return self.labels
        
def process_image2(img, tracker):
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    labels = tracker.find_vehicles(img)
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(draw_img, labels)
    return draw_img
        
def process_video_clip(clip, tracker):
    def process_frame(image):
        return process_image2(image, tracker)
    return clip.fl_image(process_frame) #NOTE: this function expects color images!!
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: vehicke_tracker.py video_path svc_pickel_path parameter_file_path\n")
        sys.exit(1)

    # load a pe-trained svc model from a serialized (pickle) file
    clf_pickle = pickle.load( open(sys.argv[2], "rb" ) )

    # get attributes of our svc object
    svc = clf_pickle["clf"]
    X_scaler = clf_pickle["scaler"]

    Input_video = sys.argv[1]
    video_file = Input_video.split('/')[-1]
    Output_video = video_file.split('.')[0]+'_output.mp4'

    with open(sys.argv[3]) as f:
        params = yaml.load(f)

    feature_params = params['feature_params']
    tracker_params = params['tracker_params']

    # Set up the overall class to do all the tracking
    vehicles = VehicleTracker(tracker_params['regions'], tracker_params['scales'], tracker_params['cells_per_step'], svc, X_scaler, feature_params, tracker_params['smooth_factor'], tracker_params['threshold'], tracker_params['key_frame_interval'])

    clip1 = VideoFileClip(Input_video)
    video_clip = clip1.fx(process_video_clip,vehicles)
    video_clip.write_videofile(Output_video, audio=False)