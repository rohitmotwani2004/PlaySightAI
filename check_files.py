import os

print("--- CURRENT FOLDER CONTENTS ---")
print(f"Scanning folder: {os.getcwd()}")
files = os.listdir()

found_video = False
for f in files:
    print(f"📄 Found: {f}")
    if "badminton" in f.lower():
        found_video = True
        print(f"   ^^^ IS THIS YOUR VIDEO? Copy this name exactly!")

if not found_video:
    print("\n❌ CRITICAL ERROR: No video file found in this folder.")
    print("Please move the video file here.")