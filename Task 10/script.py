import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper function to draw bounding boxes
def draw_bounding_box(img, bbox):
    x, y, w, h = [int(v) for v in bbox]
    return cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Load video
video_path = 'cars.mp4'
video = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = video.read()
if not ret:
    print("Failed to read video")
    exit()

# Select bounding box
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)

# If bbox is not selected, exit
if bbox == (0,0,0,0):
    print("No bounding box selected, exiting")
    exit()

cv2.destroyAllWindows()

# Initialize KCF tracker
tracker_kcf = cv2.TrackerKCF_create()
tracker_kcf.init(frame, bbox)

# Initialize CSRT tracker
tracker_csrt = cv2.TrackerCSRT_create()
tracker_csrt.init(frame, bbox)

def run_tracker(video, tracker, tracker_name, num_frames=15):
    frames = []
    for i in range(num_frames):
        ret, frame = video.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if ok:
            frame = draw_bounding_box(frame, bbox)
        else:
            cv2.putText(frame, f"{tracker_name} tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        frames.append(frame)
        
        # Save each frame with bounding box
        cv2.imwrite(f"{tracker_name}_frame_{i}.png", frame)
    
    return frames

# Reset video to the first frame
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Run KCF tracker
kcf_frames = run_tracker(video, tracker_kcf, "KCF")

# Reset video to the first frame
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Run CSRT tracker
csrt_frames = run_tracker(video, tracker_csrt, "CSRT")

# Display results for KCF and CSRT
for i in range(min(len(kcf_frames), len(csrt_frames))):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(kcf_frames[i], cv2.COLOR_BGR2RGB))
    plt.title('KCF Tracker')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(csrt_frames[i], cv2.COLOR_BGR2RGB))
    plt.title('CSRT Tracker')

    plt.show()
