import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO # type: ignore

# Function to calculate Euclidean distance
def euclidean_distance_float(p1, p2):
    return np.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)

# Function to match circles between frames based on nearest distances
def match_circles_between_frames(previous_circles, current_circles):
    matched_circles = {}
    used_indices = set()

    # Iterate over previously detected circles
    for idx, prev_circle in previous_circles.items():
        prev_center = prev_circle['center']
        min_distance = float('inf')
        match_idx = -1

        # Find the closest match in current circles
        for i, current_circle in enumerate(current_circles):
            if i in used_indices:
                continue

            curr_center = (current_circle[0], current_circle[1])
            distance = euclidean_distance_float(prev_center, curr_center)

            if distance < min_distance:
                min_distance = distance
                match_idx = i

        # If a match is found within a reasonable distance, add it to matched circles
        if match_idx != -1 and min_distance <= previous_circles[idx]['radius']:
            matched_circles[idx] = {
                'center': (current_circles[match_idx][0], current_circles[match_idx][1]),
                'radius': current_circles[match_idx][2]
            }
            used_indices.add(match_idx)

    # # Handle new circles not matched in previous frame
    # unmatched_current = [i for i in range(len(current_circles)) if i not in used_indices]
    # new_circle_id = max(previous_circles.keys(), default=-1) + 1
    # for i in unmatched_current:
    #     matched_circles[new_circle_id] = {
    #         'center': (current_circles[i][0], current_circles[i][1]),
    #         'radius': current_circles[i][2]
    #     }
    #     new_circle_id += 1

    return matched_circles

# Main function to detect and track circles in the video using YOLOv8
def detect_and_track_circles(video_path, model_path, output_csv='circle_tracking_output.csv'):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {frame_count} frames, {fps} FPS, Resolution: {width}x{height}")

    # Create a window to display tracking
    cv2.namedWindow("Circle Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Circle Tracking", width, height)

    # Read the first frame to initialize circles
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Detect circles in the first frame using YOLOv8
    results = model.predict(frame, conf=0.22, imgsz=1200, max_det=3400)
    initial_circles = [
        (int(box.xywh[0][0]), int(box.xywh[0][1]), int(box.xywh[0][3] / 2))
        for box in results[0].boxes
    ]

    if len(initial_circles) == 0:
        print("No circles detected in the first frame.")
        return

    # Initialize circle data and trajectories
    circle_data = {
        i: {'center': (circle[0], circle[1]), 'radius': circle[2], 'initial_center': (circle[0], circle[1])}
        for i, circle in enumerate(initial_circles)
    }
    circle_trajectories = {i: [(circle[0], circle[1])] for i, circle in enumerate(initial_circles)}

    # Process each frame to track circles
    for frame_num in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_num}")
            break

        # Detect circles in the current frame using YOLOv8
        results = model.predict(frame, conf=0.22, imgsz=1200, max_det=3400)
        new_circles = [
            (int(box.xywh[0][0]), int(box.xywh[0][1]), int(box.xywh[0][3] / 2))
            for box in results[0].boxes
        ]

        if new_circles:
            # Match the detected circles with previous frame's circles
            matched_circles = match_circles_between_frames(circle_data, new_circles)

            # Update circle data and trajectories based on matched circles
            for circle_id, circle in matched_circles.items():
                # Update circle_data with the matched circle's new position
                circle_data[circle_id] = circle

                # # Initialize trajectory for new circle IDs
                # if circle_id not in circle_trajectories:
                #     circle_trajectories[circle_id] = []

                # Append the new position to the circle's trajectory
                circle_trajectories[circle_id].append(circle['center'])

            # Track unmatched circles (if any)
            # for circle_id in circle_data:
            #     if circle_id not in matched_circles:
            #         # Initialize trajectory if this circle ID is new to prevent KeyError
            #         # if circle_id not in circle_trajectories:
            #         #     circle_trajectories[circle_id] = []

            #         # Append the existing position if no new position is found in this frame
            #         circle_trajectories[circle_id].append(circle_data[circle_id]['center'])

        # Visualize tracking (optional)
        for circle_id, circle in circle_data.items():
            cv2.circle(frame, circle['center'], circle['radius'], (0, 255, 0), 1)

        cv2.imshow("Circle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{frame_count} frames")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Write tracking results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Circle ID', 'Initial X', 'Initial Y', 'Final X', 'Final Y', 'Total Displacement'])

        for circle_id, trajectory in circle_trajectories.items():
            initial_pos = trajectory[0]
            final_pos = trajectory[-1]
            displacement = euclidean_distance_float(initial_pos, final_pos)
            trajectory_str = ';'.join([f"{pos[0]},{pos[1]}" for pos in trajectory])
            csvwriter.writerow([circle_id, initial_pos[0], initial_pos[1], final_pos[0], final_pos[1], displacement, trajectory_str])

    print(f"Circle tracking results have been saved to {output_csv}")

if __name__ == "__main__":
    video_path = r'VideoMov.mp4'  
    model_path = r'yolov8\runs\detect\circle_detection_yolov832\weights\best.pt'
    start_time = time.time()
    detect_and_track_circles(video_path, model_path)
    end_time = time.time()
    print(f'Total time taken to run circle detection: {end_time - start_time:.2f} seconds')
