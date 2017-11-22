import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os

########## define global parameters ########## 

#Read in a test image
test_img = mpimg.imread('test_images/test3.jpg')
img_size = (test_img.shape[1], test_img.shape[0])
# hardcode the source and destination points
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    
# Choose a Sobel kernel size    
ksize = 15 # Choose a larger odd number to smooth gradient measurements

# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

# To be replaced with camera calibration later
mtx  = None
dist = None

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Debug file
fout = None

# Keep last good lane detection to reuse when the next lane
# detection fails
last_good_left_fit = None
last_good_right_fit = None
last_good_left_curv = None
last_good_right_curv = None
last_good_offset_m = None
bad_frames = 0

left_fit = None
right_fit = None

########## definition of functions ##########        
        
def undistort_img (img, display=False) :  
    # Undistorting a calibration image:
    #img = mpimg.imread('camera_cal/calibration1.jpg')    
    global mtx
    global dist
    if mtx is None :
        tmp  = pickle.load( open("camera_cal.p", "rb") )
        #tmp  = pickle.load( open("camera_calibration_mtx_dist.p", "rb") )
        mtx  = tmp["mtx"]
        dist = tmp["dist"]
        
    undist = cv2.undistort(img, mtx, dist, None, mtx)    
    #cv2.imwrite('undist_images/undist.jpg', undist)
    #cv2.imwrite('undist_images/original.jpg', img)
    
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
    
    return undist
    
def apply_thresholds( img ) :
  s_thresh=(90, 255)
  sx_thresh=(20, 100)

  # Convert to HLS color space and separate the S channel
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]

  # Grayscale image
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # Sobel x
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
  abs_sobelx = np.absolute(sobelx)
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

  # Threshold color channel
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

  #color_binary = np.dstack(( np.zeros_like(sxbinary), 255*sxbinary, 255*s_binary))
  #plt.imshow(color_binary)
  #plt.show()

  combined = np.zeros_like(gray)
  combined[(sxbinary == 1) | (s_binary == 1)] = 1

  return combined

# Define perspective transform function
def warper(img, src, dst):
    # Define calibration box in source and destination coordinates
    img_size = (img.shape[1], img.shape[0])
   
    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)

    # Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp an image using the perspective transform, M:
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, Minv
    
def sliding_window (binimg_warped, display=False) :
    global left_fit
    global right_fit
    global bad_frames
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binimg_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binimg_warped[binimg_warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binimg_warped, binimg_warped, binimg_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print (midpoint, leftx_base, rightx_base) # 640 365 981


    # Set height of windows
    window_height = np.int(binimg_warped.shape[0]/nwindows)

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    if (left_fit is None or right_fit is None or bad_frames > 10):
        # Search window from scratch
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binimg_warped.shape[0] - (window+1)*window_height
            win_y_high = binimg_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            #print (win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high)

            # Draw the windows on the visualization image, with color green
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
    else: #Skip the sliding windows step once you know where the lines are
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) 
                        & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) 
                        & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))                

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the radius of curvature 
    ploty = np.linspace(0, binimg_warped.shape[0]-1, binimg_warped.shape[0] )
    y_eval = np.max(ploty)
    left_curv = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curv = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curv, 'm', right_curv, 'm')
    
    # Calculate the offset of vehicle to the lane center  
    offset = binimg_warped.shape[1]/2 - (rightx_base + leftx_base)/2
    offset_m = offset * xm_per_pix
    
    if display:
        # Generate x and y values for plotting
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # plot left lane Red
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # plot right lane blue
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit, left_curv, right_curv, offset_m


def check_lane_fit( left_fit, right_fit, left_curv, right_curv, offset_m ) :
    global last_good_left_fit
    global last_good_right_fit
    global last_good_left_curv
    global last_good_right_curv
    global last_good_offset_m
    global bad_frames

    global fout
    if ( fout is None ) :
        fout = open( 'debug_fit.txt', 'wt' )

    fout.write( str(left_fit[2])+' '+str(left_fit[1])+' '+str(left_fit[0])+' ' )
    fout.write( str(right_fit[2])+' '+str(right_fit[1])+' '+str(right_fit[0])+' ' )
    fout.write( str(left_curv)+' '+str(right_curv)+' ' )
    fout.write( str(offset_m)+'\n' )

    # !!!!!!!!!!!!!
    # SHOULD CHECK THAT LAST_GOOD_x IS NOT NONE FIRST
    # !!!!!!!!!!!!!
    if ( np.absolute(left_fit[2] - right_fit[2]) < 400.0
      or np.absolute(left_fit[2] - right_fit[2]) > 800.0
      or np.absolute(left_fit[1] - right_fit[1]) > 1.0
      or np.absolute(left_fit[0] - right_fit[0]) > 1e-3
       ) :
      left_fit = last_good_left_fit
      right_fit = last_good_right_fit
      left_curv = last_good_left_curv
      right_curv = last_good_right_curv
      offset_m = last_good_offset_m
      bad_frames += 1
    else :
      last_good_left_fit = left_fit
      last_good_right_fit = right_fit
      last_good_left_curv = left_curv
      last_good_right_curv = right_curv
      last_good_offset_m = offset_m
      bad_frames = 0

    return left_fit, right_fit, left_curv, right_curv, offset_m


def draw_lane( img, Minv, l_pfit, r_pfit, left_curv, right_curv, offset_m ) :
    # Create an image to draw the lines on
    lns = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx  = l_pfit[0]*ploty**2 + l_pfit[1]*ploty + l_pfit[2]
    right_fitx = r_pfit[0]*ploty**2 + r_pfit[1]*ploty + r_pfit[2]
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lns, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(lns, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    curvature = np.int(np.mean([left_curv, right_curv]))
    cv2.putText(result,'Radius of Curvature = {:5d} (m)'.format(curvature), (50,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(255,255,255),2,cv2.LINE_AA)
    if np.absolute(offset_m) > .50:
        color = (255, 0, 0) # red
    elif np.abs(offset_m) > .35:
        color = (255, 255, 0) # yellow
    else:
        color = (255, 255, 255) # white
    if offset_m > 0:    
        cv2.putText(result,'Vehicle is {:.2f} (m) right of center'.format(offset_m), (50,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,color,2,cv2.LINE_AA)
    else:
        cv2.putText(result,'Vehicle is {:.2f} (m) left of center'.format(-offset_m), (50,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,color,2,cv2.LINE_AA)
    #plt.imshow(result)
    #plt.title('Original (undistorted) image with lane area drawn')
    #plt.show()
    return result


def test_images ():
    for fname in os.listdir("test_images/"): 
        print('input file name ', fname)
        # Read in the image
        in_image = mpimg.imread('test_images/'+fname)
        out_image = process_image(in_image)
        #Make copies into the test_images_output directory
        #mpimg.imsave('test_images_output/'+fname, out_imag))


def test_videos (vnumber=0, subclip=False):
    from moviepy.editor import VideoFileClip

    if (vnumber == 0) :
        if subclip:
            clip1 = VideoFileClip("project_video.mp4").subclip(23,26)
            #clip1 = VideoFileClip("project_video.mp4").subclip(38,41)
            project_output = 'test_video_output/project_video_sub_output.mp4'
        else:
            clip1 = VideoFileClip("project_video.mp4")
            project_output = 'test_video_output/project_video_output.mp4'
        project_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
        project_clip.write_videofile(project_output, audio=False)

    if (vnumber == 1) :
        if subclip:
            clip1 = VideoFileClip("challenge_video.mp4").subclip(0,5)
            challenge_output = 'test_video_output/challenge_video_sub_output.mp4'
        else:
            clip1 = VideoFileClip("challenge_video.mp4")
            challenge_output = 'test_video_output/challenge_video_output.mp4'
        challenge_clip = clip1.fl_image(process_image) 
        challenge_clip.write_videofile(challenge_output, audio=False)
    
    if (vnumber == 2) :
        if subclip:
            clip1 = VideoFileClip("harder_challenge_video.mp4").subclip(0,5)
            harder_challenge_output = 'test_video_output/harder_challenge_video_sub_output.mp4'
        else:
            clip1 = VideoFileClip("harder_challenge_video.mp4")    
            harder_challenge_output = 'test_video_output/harder_challenge_video_output.mp4'            
        harder_challenge_clip = clip1.fl_image(process_image) 
        harder_challenge_clip.write_videofile(harder_challenge_output, audio=False)
    

########## Pipeline ########## 
def process_image(img):
    ##1. Undistort the test image using mtx and dist
    undist = undistort_img (img)
    #plt.imshow(undist)
    #plt.title('undistort_test_image')
    #plt.show()

    ##2. Use color transforms, gradients, etc., to create a thresholded binary image
    combined_binary = apply_thresholds( undist )
    #plt.imshow(combined_binary, combined_binary='gray')
    #plt.show()

    ##3. Apply a perspective transform to a thresholded binary image, 
    thresholded_binary = np.dstack(( combined_binary, combined_binary, combined_binary)) * 255
    binary_warped, Minv = warper(thresholded_binary, src, dst)

    ##4. Detect lane pixels and fit to find the lane boundary.
    wrp = cv2.cvtColor(binary_warped, cv2.COLOR_RGB2GRAY)
    l_pfit, r_pfit, left_curv, right_curv, offset_m = sliding_window( wrp )

    ##5. Sanity check
    l_pfit, r_pfit, left_curv, right_curv, offset_m = check_lane_fit( l_pfit, r_pfit, left_curv, right_curv, offset_m )
    
    ##6. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
    # Create an image to draw the lines on
    lane_display = draw_lane( undist, Minv, l_pfit, r_pfit, left_curv, right_curv, offset_m )

    return lane_display
    

#test_images()

test_videos(0, False)


