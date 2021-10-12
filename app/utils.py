import cv2


def prepare_output_video_stream(input_video_stream: cv2.VideoCapture,
                                output_video_file_path: str) -> cv2.VideoWriter:
    fps = int(input_video_stream.get(cv2.CAP_PROP_FPS))
    width = int(input_video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_video_file_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
    return video_writer
