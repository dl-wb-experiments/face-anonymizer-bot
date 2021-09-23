import cv2
import numpy as np


def blur(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    pixels_count = 16

    # Resize the image to pixels_count*pixels_count with interpolation to blur the image
    resized_image = cv2.resize(image, (pixels_count, pixels_count), interpolation=cv2.INTER_LINEAR)
    # Resize the image to original image size
    return cv2.resize(resized_image, (width, height), interpolation=cv2.INTER_NEAREST)


def prepare_output_video_stream(input_video_stream: cv2.VideoCapture,
                                output_video_file_path: str) -> cv2.VideoWriter:
    fps = 20
    width = int(input_video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_video_file_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    return video_writer
