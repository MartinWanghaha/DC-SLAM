import os
import gdown


def download_models():
    os.makedirs('weights', exist_ok=True)

    # DROID-SLAM weights
    print("Downloading DROID-SLAM weights...")
    gdown.download(
        "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view",
        output="weights/droid.pth",
        fuzzy=True
    )

    # DepthPro weights
    # Download from Apple's ml-depth-pro release
    print("Downloading DepthPro weights...")
    print("Please download DepthPro weights manually from:")
    print("  https://github.com/apple/ml-depth-pro")
    print("Place the checkpoint at: weights/depth_pro.pt")


if __name__ == "__main__":
    download_models()
