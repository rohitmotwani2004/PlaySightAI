from ultralytics import YOLO
import cv2
import sys

# 1. Setup Paths
video_path = "badminton_match.mp4"
court_model_path = "models/yolov8-court.pt" # Check if this path is correct for you!
player_model_path = "yolov8n.pt"

print(f"--- DEBUGGING START ---")

# 2. Load Video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video '{video_path}'. Check filename!")
    sys.exit()

# 3. Skip to frame 100 (to ensure we are in the game, not the intro)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
if not ret:
    print("❌ ERROR: Could not read frame 100. Video might be too short.")
    sys.exit()

print("✅ Video loaded successfully.")

# 4. TEST: Draw a Big Blue Circle (To prove drawing works)
# If you don't see this circle in the output, your OpenCV is broken.
cv2.circle(frame, (100, 100), 50, (255, 0, 0), -1) 
print("✅ Drew Test Circle (Blue) at 100,100")

# 5. TEST: Player Detection
print("\n--- TESTING PLAYERS ---")
try:
    player_model = YOLO(player_model_path)
    # Extremely low confidence (0.1) just to find SOMETHING
    player_results = player_model(frame, conf=0.1)[0]
    
    count = len(player_results.boxes)
    print(f"found {count} potential players.")
    
    for box in player_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # Draw Green Box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
except Exception as e:
    print(f"❌ PLAYER MODEL FAILED: {e}")

# 6. TEST: Court Detection
print("\n--- TESTING COURT ---")
try:
    # Check if file exists first
    import os
    if not os.path.exists(court_model_path):
        # Try root folder if 'models/' fails
        if os.path.exists("yolov8-court.pt"):
            court_model_path = "yolov8-court.pt"
            print(f"⚠️ Found model in root folder, switching path...")
        else:
            print(f"❌ ERROR: Cannot find 'yolov8-court.pt'. Put it in the folder!")

    court_model = YOLO(court_model_path)
    # Extremely low confidence (0.1)
    court_results = court_model(frame, conf=0.1)[0]
    
    if court_results.keypoints is not None:
        kpts = court_results.keypoints.xy.cpu().numpy()[0]
        print(f"Found {len(kpts)} keypoints.")
        for x, y in kpts:
            if x > 0 and y > 0:
                # Draw Red Dot
                cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
    else:
        print("Found 0 keypoints (Model saw nothing).")

except Exception as e:
    print(f"❌ COURT MODEL FAILED: {e}")

# 7. Save Result
output_filename = "debug_result.jpg"
cv2.imwrite(output_filename, frame)
print(f"\n✅ DONE! Open '{output_filename}' to see the result.")
print("--- DEBUGGING END ---")