import cv2 as cv
import kps as kps
import numpy as np
import json
import torch
from LK import Estimator
import argparse
import features

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--features", help="Please specify features capture method")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("/workspace/config/lk.json", "r") as lk_file:
    lk_params = json.load(lk_file)

with open("/workspace/config/corner.json", "r") as corner_file:
    corner_params = json.load(corner_file)

arrow_length = 1
# Variable for color to draw optical flow track
color = (0, 255, 0)
color_line = (0, 0, 255)

# specify the input video here
input_path = "/workspace/data/self/library-busy.mp4"
# specify the output video here
output_path = "/workspace/output/test5.mp4"
output_fps = 0.5
threshold=1e-2
if __name__ == '__main__':
    cap = cv.VideoCapture(input_path)

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # These features would be tracked in the following optical flow tracking pipeline
    if args.features == 'corner':
        # Finds the strongest corners in the first frame by Shi-Tomasi method
        prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)
    if args.features == 'sift':
        # Finds the keypoints by SIFT
        key_points = features.getKeypointsAndDescriptorsBySIFT(prev_gray)
        prev = np.around(np.array(list(map(lambda point: point.pt, key_points)))).astype(np.float32).reshape(-1, 1, 2)

    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    est = Estimator(win_size=15, tau=threshold)
    frame_counter = 0
    while (cap.isOpened()):
        # frame is the current frame being projected in the video
        ret, frame = cap.read()
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        if args.features == 'corner':
            prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)
        if args.features == 'sift':
            key_points = features.getKeypointsAndDescriptorsBySIFT(prev_gray)
            prev = np.around(np.array(list(map(lambda point: point.pt, key_points)))).astype(np.float32).reshape(-1, 1, 2)

        feature_pos_x = prev[:, 0, 0].astype(int)
        feature_pos_y = prev[:, 0, 1].astype(int)

        movement, is_features_updated = est.lk(prev_gray, gray, prev)
        # Selects moving feature points for previous position
        feature_points_old = prev.astype(int)
        # Calculate feature positions for next position
        moving_features = prev[is_features_updated.cpu().numpy() == 1].astype(int)
        movement_array = movement.cpu().numpy()
        feature_pos_x_new = feature_pos_x + movement_array[:, 0]
        feature_pos_y_new = feature_pos_y + movement_array[:, 1]
        features_movement = np.around(np.stack((feature_pos_x_new, feature_pos_y_new), axis=1)).reshape(-1, 1, 2)[
            is_features_updated.cpu().numpy() == 1].astype(int)

        # visualize feature points : the optical flow tracks
        for i, (new, old) in enumerate(zip(features_movement, moving_features)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            direction_x = c - a
            direction_y = d - b
            extended_dest_x = a + direction_x * arrow_length
            extended_dest_y = b + direction_y * arrow_length
            # Draws line between new and old position with green color and 2 thickness
            mask = cv.arrowedLine(mask, (a, b), (extended_dest_x, extended_dest_y), color_line, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv.add(frame, mask)

        frame_counter += 1

        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = features_movement.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        cv.imshow("sparse optical flow", output)

    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
