import os
import cv2
import glob
import shutil

def extract_frames(video_path, output_folder, frame_interval=1, max_frame=None):
    """
    Extract frames from an MP4 video and save them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Directory where extracted images will be saved.
    - frame_interval: Save every nth frame (default is 1, meaning save all frames).
    - max_frame: Maximum number of frames to save (default is 8).
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        if (max_frame is not None) and (saved_count >= max_frame):
            break  # End of video or reached max_frame

        # Save every nth frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    # print(f"Extracted {saved_count} frames and saved to {output_folder}")


def extract_images(images_path, output_folder, frame_interval=1, max_frame=None):
    """
    Select frames from an image folder and save them to output_folder.

    Parameters:
    - images_path: Path to the folder containing frame images.
    - output_folder: Directory where selected images will be saved.
    - frame_interval: Pick every nth image (default is 1, meaning pick all).
    - max_frame: Maximum number of images to save (default is 8).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get and sort image files
    image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))

    selected_count = 0
    for idx, img_path in enumerate(image_files):
        if idx % frame_interval == 0:
            # Copy image to output folder
            output_path = os.path.join(output_folder, f"frame_{selected_count:04d}.png")
            shutil.copy2(img_path, output_path)
            selected_count += 1

            if max_frame is not None:
                if selected_count >= max_frame:
                    break

    print(f"Copied {selected_count} frames to {output_folder}")
