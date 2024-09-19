import os
from moviepy.editor import VideoFileClip
import math

def compress_video(input_path, output_path, target_size_mb=10, target_fps=None, target_resolution=None):
    """
    Compresses a video to a target file size.

    :param input_path: Path to the input video file.
    :param output_path: Path where the compressed video will be saved.
    :param target_size_mb: Maximum desired file size in megabytes.
    :param target_fps: (Optional) Target frames per second.
    :param target_resolution: (Optional) Target resolution as (width, height).
    """
    # Convert target size from MB to bits
    target_size_bits = target_size_mb * 8 * 10**6  # 10 MB = 80,000,000 bits

    # Load the video clip
    clip = VideoFileClip(input_path)
    duration = clip.duration  # in seconds

    # Estimate audio bitrate (let's assume 128 kbps)
    audio_bitrate = 128000  # in bits per second

    # Calculate the maximum allowed video bitrate
    max_video_bitrate = (target_size_bits / duration) - audio_bitrate
    if max_video_bitrate <= 0:
        raise ValueError("Target size too small to accommodate audio bitrate.")

    # Convert bitrate to kilobits per second for ffmpeg
    max_video_bitrate_k = math.floor(max_video_bitrate / 1000)

    print(f"Video Duration: {duration:.2f} seconds")
    print(f"Target File Size: {target_size_mb} MB")
    print(f"Calculated Video Bitrate: {max_video_bitrate_k} kbps")

    # Optional: Resize the video to reduce bitrate requirements
    if target_resolution:
        clip = clip.resize(newsize=target_resolution)
        print(f"Resized video to: {target_resolution}")
    elif max_video_bitrate_k < 500:
        # If bitrate is very low, consider reducing resolution to help
        new_width = 640
        new_height = int(new_width * clip.h / clip.w)
        clip = clip.resize(newsize=(new_width, new_height))
        print(f"Auto-resized video to: {(new_width, new_height)} to accommodate bitrate constraints.")

    # Set fps if specified
    if target_fps:
        clip = clip.set_fps(target_fps)
        print(f"Set FPS to: {target_fps}")

    # Export the video with the calculated bitrate
    # Note: moviepy's write_videofile uses ffmpeg under the hood
    clip.write_videofile(
        output_path,
        bitrate=f"{max_video_bitrate_k}k",
        audio_bitrate="128k",
        codec="libx264",
        preset="medium",
        threads=4,
        verbose=True,
        logger=None  # Set to 'bar' or 'full' for more verbose output
    )

    # Verify the output file size
    output_size = os.path.getsize(output_path) / (8 * 10**6)  # Convert to MB
    print(f"Output File Size: {output_size:.2f} MB")

    if output_size > target_size_mb:
        print("Warning: The output file size exceeds the target size. Consider lowering the resolution or bitrate further.")
    else:
        print("Success: The video has been compressed to the target size.")

if __name__ == "__main__":
    input_video_path = r"C:\Users\tjerf\source\repos\RifeTensorRT\Output2.mp4"
    output_video_path = r"C:\Users\tjerf\source\repos\RifeTensorRT\Output2_compressed.mp4"
    try:
        compress_video(
            input_path=input_video_path,
            output_path=output_video_path,
            target_size_mb=10,
            target_fps=48,  # Optional: Set desired FPS
            target_resolution=(640, 360)  # Optional: Set desired resolution
        )
    except Exception as e:
        print(f"An error occurred: {e}")
