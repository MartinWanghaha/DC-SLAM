import os
import cv2
import shutil

SEQ_NUM = 15
STEP = 10


def download_dataset():
    """Download ADVIO example dataset."""
    try:
        import py3_wget
    except ImportError:
        print("py3_wget not installed. Install with: pip install py3_wget")
        print("Or download manually from: https://zenodo.org/record/1476931")
        return

    os.makedirs("data", exist_ok=True)
    py3_wget.download_file(
        f"https://zenodo.org/record/1476931/files/advio-{SEQ_NUM}.zip",
        output_path=f"data/advio-{SEQ_NUM}.zip",
    )
    shutil.unpack_archive(f"data/advio-{SEQ_NUM}.zip", "data/")

    os.makedirs(f"data/advio-{SEQ_NUM}/iphone/frames", exist_ok=True)
    video_cap = cv2.VideoCapture(f"data/advio-{SEQ_NUM}/iphone/frames.mov")

    tot_num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for num_frame in range(0, tot_num_frames, STEP):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, num_frame - 1)
        success, image = video_cap.read()
        if success:
            cv2.imwrite(f"data/advio-{SEQ_NUM}/iphone/frames/frame_{num_frame:04d}.jpg", image)

    with open(f"data/advio-{SEQ_NUM}/iphone/intrinsics.txt", "w") as file:
        file.write("1082.4\n1084.4\n364.68\n643.31\n")


if __name__ == "__main__":
    download_dataset()
