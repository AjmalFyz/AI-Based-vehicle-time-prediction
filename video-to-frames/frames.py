import cv2
import os


def convert_video_to_frames(video_path, output_dir):
    # Read the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Create an output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If the frame is not read properly, it means we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = f"{output_dir}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

        # Display the frame count and the current frame
        print(f"Processed frame {frame_count}")

        # Increment the frame count
        frame_count += 1

    # Release the video object
    video.release()

    print("Finished converting video to frames")


# Example usage
video_path = "./video.mp4"
output_dir = "./out"
convert_video_to_frames(video_path, output_dir)
