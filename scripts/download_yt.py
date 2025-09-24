import sys
from eye.youtube_utils import download_360_video

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_360.py <youtube_url> <output_folder>")
        sys.exit(1)

    url = sys.argv[1]
    output_folder = sys.argv[2]
    download_360_video(url, output_folder)
