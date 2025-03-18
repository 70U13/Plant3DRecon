# IMPORT DEPENDENCIES
import os
import torch
import numpy as np
import cv2 as cv
import open3d as o3d

from matplotlib import pyplot as plt
# pip install timm (for MiDaS model)

'''
1  Change the path of the folder containing the ff.:
    a. calibration images
    b. stereo images
2  Change the substrings to match the images in the directory, either of the ff.:
    a. '_LC' and '_RC'
    b. 'leftcam' and 'rightcam'
'''

# GLOBAL VARIABLES
cam_calibration_dir_path = os.path.abspath("./GEN_3D_Plants/calibration_images_best")

if not os.path.exists(cam_calibration_dir_path):
    print(f"Error: Calibration directory does not exist! ({cam_calibration_dir_path})")

stereo_images_dir_path = "C:\\Users\\Kaye Louise\\Desktop\\research-3D\\GEN_3D_Plants\\sample_traditional-3"

processed_images_dir_path = "./processed"

# Open3D Translate Function Multiplier
X = 4000.0      

# FUNCTION DEFINITIONS
def get_image_pairs(dir_path: str, left_substr="_LC", right_substr="_RC"): 
    '''Retrieve one or more image pairs (in '.jpg' or '.png' format) from the target directory based on their substring keywords (ex. \"imageX\"), 
       and each image has a file name and image data.
    '''
    image_pairs = []

    for file in os.listdir(dir_path):
        if (left_substr in file) and (file.endswith(".jpg") or file.endswith(".png")):
            img1 = [[file, cv.imread(os.path.join(dir_path, file))], None]
            image_pairs.append(img1)

    for img1 in image_pairs:
        image_name = img1[0][0]
        img2 = image_name.replace(left_substr, right_substr)
        for file in os.listdir(dir_path):  
            if file == img2:
                img1[1] = [file, cv.imread(os.path.join(dir_path, file))]
    
    return image_pairs

def get_calib_images(dir_path, additional_path=""):
    '''Retrieve one or more calibration images (in '.jpg' or '.png' format) from the target directory, specifically its file name and image data.\n
       Optional Args: additional_path = add a more specific path to the main directory 'dir_path'
    '''
    
    images = []

    # Retrieve all files inside the directory
    file_path = os.path.join(dir_path, additional_path)
    print(f" >>> Checking if directory exists: \"{os.path.abspath(file_path)}\"")
    
    if not os.path.exists(file_path):
        print(" >>> Directory does not exist!")
        return []

    # Loop through all files in the directory
    for file_name in os.listdir(file_path):
        if file_name.lower().endswith((".jpg", ".png")):  # Correct file format check
            img_path = os.path.join(file_path, file_name)
            img = cv.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f" >>> Warning: Could not read {img_path}")

    print(f" >>> Found {len(images)} images for calibration...")
    return images

def calibrate_stereo_cam(cam_calibration_dir_path, chessboard_size=(10,7), frame_size=(640, 480)):
    '''Generate stereo camera calibration parameters from a set of calibration images for each camera\n
       Optional Args: chessboard_size = specify the number of corners between the first and second row, and between the first and 
                                                             second columns, respectively
                               frame_size = specify the width and height of the calibration images (must be uniform for all calibration images)
    
    '''
    # Declare constants & initialize variables
    SIZE_OF_CHESSBOARD_SQUARES_MM = 125 #20  # Size of each Chessboard Square (mm)
    images_left = []                    # Images to be used for Calibration (left CCamera)
    images_right = []                   # Images to be used for Calibration (Right Camera)
    obj_points = []                     # 3D Point in Real World Space
    img_points_left = []                # 2D Points in Image Plane (Left Camera)
    img_points_right = []               # 2D Points in Image Plane (Right Camera)
    
    # Define Termination Criteria; If number of iterations=30 reached or error is less than epsilon=0.001, calibration is terminated
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare Object Points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * SIZE_OF_CHESSBOARD_SQUARES_MM

    # Retrieve Images for Calibration
    images_left = get_calib_images(cam_calibration_dir_path, "left_cam")     # Retrieve Images For Left Camera
    images_right = get_calib_images(cam_calibration_dir_path, "right_cam")   # Retrieve Images For Right Camera

    # Find the Chessboard Corners for Left Camera & Right Camera Images
    print(" >>> Detecting Chessboard Corners...")

    img_count = 0
    for img_left, img_right in zip(images_left, images_right):
        # For each image pair, convert to grayscale for better results
        imgL = img_left
        imgR = img_right
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the Chessboard Corners; If found, append object points and image points to respective lists
        retL, cornersL = cv.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboard_size, None)

        if retL and retR == True:
            obj_points.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            img_points_left.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            img_points_right.append(cornersR)

            # Optional: Draw and Display the Corners
            #cv.drawChessboardCorners(imgL, chessboard_size, cornersL, retL)
            #cv.imshow("Left Camera: Image " + str(img_count + 1), imgL)
            #cv.drawChessboardCorners(imgR, chessboard_size, cornersR, retR)
            #cv.imshow("Right Camera: Image " + str(img_count + 1), imgR)
            #cv.waitKey(0)   # Press any key or close window directly
        
        else:
             print(f"   * ERROR: Unable to find corners for Image {img_count + 1}")
        img_count += 1

    cv.destroyAllWindows()

    # ===================== CALIBRATION ========================== #
    
    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(obj_points, img_points_left, frame_size, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(obj_points, img_points_right, frame_size, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    # ===================== STEREO VISION CALIBRATION ========================== #
    # Fix Intrinsic Camera Matrices, so that only Rot, Trns, Emat and Fmat are calculated
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Same as termination criteria above
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(obj_points, img_points_left, img_points_right, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    # Optional: Calculate Reprojection Error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv.projectPoints(obj_points[i], rvecsR[i], tvecsR[i], newCameraMatrixR, distR)
        error = cv.norm(img_points_right[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    # DEBUG: Display Total Error
    print(f" >>> Total Error: {mean_error/len(obj_points)}")

    # Rectify Stereo Camera Maps
    rectify_scale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectify_scale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    # Save Calibration Parameters in a single .xml
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('q', Q)

    cv_file.release()

def create_pcd(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

# ================================= PROGRAM START ================================== #

# ----- 1. RETRIEVE STEREO IMAGES FROM TARGET DIRECTORY INTO IMAGE PAIR DATA STRUCTURE ----- #

# DEBUG: Display process & target directory
print(f"\nRETRIEVING STEREO IMAGES FROM: \"{stereo_images_dir_path}\"")     

image_pairs_raw = get_image_pairs(stereo_images_dir_path)

# DEBUG: Display number of image pairs found
print(f" >>> Found {len(image_pairs_raw)} image pairs...")                  

# DEBUG: Display each image filename & image data
#for image_pair in image_pairs_raw:                                          
#    cv.imshow(image_pair[1][0], image_pair[1][1])
#    cv.waitKey(0)
#    cv.imshow(image_pair[0][0], image_pair[0][1])
#    cv.waitKey(0)


#  ----- 2. PERFORM CAMERA CALIBRATION TO GENERATE CAMERA PARAMETERS ----------------------- #

# DEBUG: Display process & target directory
print(f"\nRETRIEVING CALIBRATION IMAGES FROM: \"{cam_calibration_dir_path}\"")     

calibrate_stereo_cam(cam_calibration_dir_path)

# DEBUG: Display target file for retrieving camera parameters
print(f"CAMERA PARAMETERS GENERATED WITH FILENAME: \"stereoMap.xml\"")             


# ----- 3. UNDISTORT STEREO IMAGES USING GENERATED CAMERA PARAMETERS ----------------------- #

# DEBUG: Display process & number of image pairs
print(f"\nUNDISTORTING {len(image_pairs_raw)} IMAGE PAIRS:")     

cam_params = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_READ)

# DEBUG: Confirm whether camera parameters exist
if cam_params.isOpened():                                                               
    print(" >>> Using Camera Parameters with filename: \"stereoMap.xml\"")
else:
    print(" >>> No Camera Parameters found! Perform Camera Calibration first.")
    exit()

# Get parameters from each node
stereoMapL_x = cam_params.getNode("stereoMapL_x").mat()
stereoMapL_y = cam_params.getNode("stereoMapL_y").mat()
stereoMapR_x = cam_params.getNode("stereoMapR_x").mat()
stereoMapR_y = cam_params.getNode("stereoMapR_y").mat()

matrixQ = cam_params.getNode("q").mat()

image_pairs_undistorted = []
# Remap raw stereo images based on camera parameters
for imp in image_pairs_raw:
    undistortedL = cv.remap(imp[1][1], stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    undistortedR = cv.remap(imp[0][1], stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    # Save undistorted image pairs to new variable
    imp_undistorted = [[imp[0][0], undistortedR], [imp[1][0], undistortedL]]
    image_pairs_undistorted.append(imp_undistorted)

# DEBUG: Display number of undistorted image pairs
print(f"PERFORMED UNDISTORTION ON {len(image_pairs_undistorted)} IMAGE PAIRS")  

# DEBUG: Display each undistorted image filename & image data
# for image_pair in image_pairs_undistorted:                                             
#     cv.imshow(image_pair[1][0], image_pair[1][1])
#     cv.waitKey(0)
#     cv.imshow(image_pair[0][0], image_pair[0][1])
#     cv.waitKey(0)


# ----- 4. PREPROCESSING & PERFORM IMAGE SEGMENTATION TO GENERATE OBJECT MASK ------------------------------ #

# DEBUG: Display process & number of image pairs
print(f"\nCREATING OBJECT MASKS FOR {len(image_pairs_undistorted)} IMAGE PAIRS:")     

image_masks = []

for im in image_pairs_undistorted:
    # Optional: Switch between im[0][1] and im[1][1] accordingly (which image will the mask be based on)
    img_raw = im[0][1]   

    # DEBUG: Show raw image (no segmentation performed)
    # cv.imshow("RAW", img_raw)
    # cv.waitKey(0)

    # Optional: Resize/Crop Image especially if large resolution, reduces processing time
    # img_raw = img_raw[0:640, 0:480]
    # img_raw = cv.resize(img_raw, None, fx=1.0, fx=1.0)

    # Define ROI: Exclude lower part where pot might be
    #height, width = img_raw.shape[:2]
    #roi = img_raw[0:int(height*0.75), :]  # Use only the top 75% of the image

    # Apply segmentation to this region only
    img_gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

    # Optional: Adjust Minimum and Maximum Threshold based on Contour results
    min_thres = 135     #105 - IMPORTANT
    max_thres = 150     #200 - IMPORTANT

    # Generate Threshold for finding image Contours     
    _, th2 = cv.threshold(img_gray, min_thres, max_thres, cv.THRESH_BINARY)

    # Find Image Contours
    contours, hierarchy = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    new_contours = []
    
    # Optional: Adjust Contour Size based on results
    contour_size = 700 #700 - IMPORTANT

    # Retrieve only Contours with a target minimum number of points (Contour Size)
    for i in range(len(contours)):
        if len(contours[i]) > contour_size:
            new_contours.append(contours[i])

    # DEBUG: Display remaining contours
    # cv.drawContours(img_raw, new_contours, -1, (0, 255, 0), thickness=2)
    # cv.imshow(im[1][0] + "Contour Result", img_raw)
    # cv.waitKey(0)

    # Create Mask from each image
    img_mask = np.zeros((img_raw.shape[0], img_raw.shape[1]), dtype="uint8")
    obj_mask = cv.drawContours(img_mask, new_contours, -1, 255, cv.FILLED)
    image_masks.append(obj_mask)

    # DEBUG: Display Mask Process Result
    # cv.imshow(im[1][0] + "Mask Result", obj_mask)
    # cv.waitKey(0)

# DEBUG: Display process & number of image masks
print(f" >>> Created {len(image_masks)} Object Masks...")     

# ----- 5. GENERATE DEPTH MAPS FROM STEREO IMAGES & APPLY OBJECT MASK ---------------------- #

# DEBUG: Display process & number of image pairs
print(f"\nCREATING DEPTH MAPS & POINT CLOUDS FOR {len(image_pairs_undistorted)} IMAGE PAIRS:")     

# Select which MiDaS model to download (MiDas_small < DPT_Hybrid < DPT_Large)
#midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")   # 
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")     # No masking
#midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")     #

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Input Transformations
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

output_points = None
output_colors = None

for im, mask in zip(image_pairs_undistorted, image_masks):
    
    # Convert image to RGB format
    img_color = cv.cvtColor(im[0][1], cv.COLOR_BGR2RGB)
    img_batch = transform(img_color).to("cpu")

    # Generate Depth Map
    with torch.no_grad():
        prediction = midas(img_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_color.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    img_depth = cv.normalize(depth_map, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    # Check depth map statistics
    #print(f"Depth Map Statistics for Current Image:")
    #print(f"Min Depth: {np.min(depth_map):.4f}")
    #print(f"Max Depth: {np.max(depth_map):.4f}")

    # DEBUG: Display MiDaS Depth Map Results
    #cv.imshow("midas prediction", img_depth)
    #cv.waitKey(0)

# ----- 6. GENERATE POINT CLOUDS FROM THE DEPTH MAPS --------------------------------------- #

    # Reproject image to 3D using Camera Parameters (Q Matrix)
    points_3D = cv.reprojectImageTo3D(img_depth, matrixQ, handleMissingValues=False)

    # Apply Mask to image before generating the Point Cloud
    mask_map = mask > 1 #0.2 - IMPORTANT
    output_points = points_3D[mask_map]
    output_colors = img_color[mask_map]

    # Display Points Shape & Colors Shape
    #print(f"Output points shape: {output_points.shape}")
    #print(f"Output colors shape: {output_colors.shape}")

    # Convert Depth Map to 8-bit Unsigned Integer Datatype & Add Color Data
    img_depth = (img_depth*255).astype(np.uint8)
    img_depth = cv.applyColorMap(img_depth , cv.COLORMAP_MAGMA)

    # Save Point Cloud with specified filename
    output_file = im[0][0].replace(im[0][0][17:], ".ply")
    create_pcd(output_points, output_colors, output_file)

    # Check depth map statistics
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    mean_depth = np.mean(depth_map)

    print(f"\nDepth Map Statistics:")
    print(f"Min Depth: {min_depth:.4f}")
    print(f"Max Depth: {max_depth:.4f}")
    print(f"Mean Depth: {mean_depth:.4f}")
    
    # Check the output file
    #print(f"Point Cloud saved to: {output_file}")  # Add this line

    # DEBUG: Display Color Map, Depth Map, and Object Mask
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(output_file)

    fig.add_subplot(1, 3, 1)
    plt.imshow(im[0][1])
    plt.axis("off")
    plt.title("Color")

    fig.add_subplot(1, 3, 2)
    plt.imshow(img_depth)
    plt.axis("off")
    plt.title("Depth")

    fig.add_subplot(1, 3, 3)
    plt.imshow(mask)
    plt.axis("off")
    plt.title("Mask")
    plt.show()
    
    # Generate Point Cloud 
    create_pcd(output_points, output_colors, output_file)

    # DEBUG: Display newly generated Point Clouds
    #pcd = o3d.io.read_point_cloud(output_file)
    #o3d.visualization.draw_geometries([pcd])

# DEBUG: Display number of depth maps & point clouds generated
print(f" >>> Created {len(image_pairs_undistorted)} Depth Maps and {len(image_pairs_undistorted)} Point Clouds...")

