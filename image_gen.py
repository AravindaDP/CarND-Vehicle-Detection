import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from pipeline import pipeline 
from tracker import LineTracker
import sys
import yaml

# ## Apply a perspective transform to rectify binary image to create a "birds-eye view"
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def map_lane(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

    return warped


# This expects undistorted RGB images
def process_image(img, src, dst, thresholds, tracker):
    result = pipeline(img, thresholds['l_thresh'], thresholds['b_thresh'], thresholds['grad_thresh'], thresholds['dir_thresh'])
    warped = warper(result,src,dst)
    
    left_line, right_line = tracker.find_lines(warped)
    
    road_img = tracker.get_road_img(warped)   
   
    road_warped = map_lane(road_img,src,dst)

    result = cv2.addWeighted(img,1.0,road_warped,0.5,0.0)

    if left_line.detected and right_line.detected:
        ym_per_pix = tracker.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = tracker.xm_per_pix # meters per pixel in x dimension

        curve_fit_cr = np.polyfit(np.array(left_line.yvals,np.float32)*ym_per_pix,np.array(left_line.bestx+right_line.bestx,np.float32)*xm_per_pix/2.0,2)
        curverad = ((1 + (2*curve_fit_cr[0]*left_line.yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])

        # calculate the offset of the car on the road
        center_diff = (left_line.line_base_pos + right_line.line_base_pos)/2
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'

        # draw the text showing curvature, offset, and speed
        cv2.putText(result, 'Radius of Curvature = '+str(int(curverad))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,2)))+'m '+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        if left_line.line_base_pos > - 0.9 or right_line.line_base_pos < 0.9: #Approx half of average width of a car
            cv2.putText(result,'Lane Departure Warning!',(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            # Force detecting new lane positions
            left_line.detected = False
            right_line.detected = False
            left_line.recent_xfitted = []
            right_line.recent_xfitted = []
            left_line.allx = [] 
            right_line.allx = [] 
            left_line.ally = [] 
            right_line.ally = []
            
    return result

    
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: image_gen.py image_path cal_image_folder_path parameter_file_path\n  note: remember to use trailing '/' in folder paths. e.g. camera_cal/")
        sys.exit(1)
        
    cal_image_path = sys.argv[2]

    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open(cal_image_path+"dist_pickle.p", "rb" ))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    image_path = sys.argv[1]
    image_file = image_path.split('/')[-1]

    # Test undistortion on an test image
    img = cv2.imread(image_path)
    img_size = (img.shape[1], img.shape[0])
    
    with open(sys.argv[3]) as f:
        config = yaml.load(f)

    dst_img = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('output_images/'+image_file.split('.')[0]+'_undist.jpg',dst_img)

    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    # Visualize undistorted image
    plt.imshow(dst_img)

    thresholds = config['thresholds']

    result = pipeline(dst_img, thresholds['l_thresh'], thresholds['b_thresh'], thresholds['grad_thresh'], thresholds['dir_thresh'])
    cv2.imwrite('output_images/binary_'+image_file,result)

    # Visualize binary image
    plt.imshow(result,cmap='gray')

    src = np.array(config['src']).astype(np.float32)
    dst = np.array(config['dst']).astype(np.float32)

    top_down = warper(dst_img,src,dst)

    cv2.polylines(dst_img, np.int32([src]), True, (255,0,0), 3)
    cv2.polylines(top_down, np.int32([dst]), True, (255,0,0), 3)

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(dst_img)
    ax1.set_title('Undistorted Image with source points drawn', fontsize=20)
    ax2.imshow(top_down)
    ax2.set_title('Warped result with dest. points drawn', fontsize=20)
    #f.savefig('output_images/warped_straight_lines.jpg')
    plt.show()

    warped = warper(result,src,dst)

    cv2.imwrite('output_images/warped_'+image_file,warped)

    # Visualize binary image
    #plt.imshow(warped, cmap='gray')
    
    tracker_params = config['tracker_params']

    # Set up the overall class to do all the tracking
    curve_centers = LineTracker(window_width = tracker_params['window_width'], window_height = tracker_params['window_height'], margin = tracker_params['margin'], ym = tracker_params['ym_per_pix'], xm = tracker_params['xm_per_pix'])
    left_line, right_line = curve_centers.find_lane_pixels(warped)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped))

    out_img[left_line.ally, left_line.allx] = [255, 0, 0]
    out_img[right_line.ally, right_line.allx] = [0, 0, 255]
    plt.imshow(out_img)


    # fit the lane boundaries to the left,right center positions found
    left_line, right_line = curve_centers.find_lines(warped)

    left_lane = np.array(list(zip(np.concatenate((left_line.bestx-2,left_line.bestx[::-1]+2),axis=0),np.concatenate((left_line.yvals,left_line.yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_line.bestx-2,right_line.bestx[::-1]+2),axis=0),np.concatenate((right_line.yvals,right_line.yvals[::-1]),axis=0))),np.int32)

    cv2.fillPoly(out_img,[left_lane],color=[0,255,0])
    cv2.fillPoly(out_img,[right_lane],color=[0,255,0])
    
    cv2.putText(out_img, 'f_right(y) = ('+str(round(right_line.current_fit[0],4))+')*y^2+('+str(round(right_line.current_fit[1],3))+')*y+('+str(round(right_line.current_fit[2],1))+')',(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(out_img, 'f_left(y) = ('+str(round(left_line.current_fit[0],4))+')*y^2+('+str(round(left_line.current_fit[1],3))+')*y+('+str(round(left_line.current_fit[2],1))+')',(50,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    # Visualize binary image with line fit
    #plt.imshow(out_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_images/'+image_file.split('.')[0] +'_fit_lines.jpg',out_img)

    # Set up the overall class to do all the tracking
    curve_centers = LineTracker(window_width = tracker_params['window_width'], window_height = tracker_params['window_height'], margin = tracker_params['margin'], ym = tracker_params['ym_per_pix'], xm = tracker_params['xm_per_pix'])

    result = process_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dist_pickle, src, dst, thresholds, curve_centers)

    # Visualize undistorted image
    #plt.imshow(result)

    result_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_images/'+image_file.split('.')[0] +'_output.jpg',result_img)
