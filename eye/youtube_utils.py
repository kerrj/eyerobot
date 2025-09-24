import yt_dlp
import sys
import json
from pathlib import Path
import subprocess

def inject_360_metadata(video_path):
    """
    Injects 360° metadata into the MP4 file using Google's spatial media metadata injector.
    Requires `spatialmedia` from https://github.com/google/spatial-media.
    """
    try:
        print("📌 Injecting 360° metadata into the video...")
        subprocess.run(["python", "-m", "spatialmedia", "-i", str(video_path)], check=True)
        print("✅ 360° metadata injection complete!")
    except Exception as e:
        print(f"⚠️ Error injecting 360° metadata: {e}")

def reencode_video(video_path):
    """
    Re-encodes the video to H.264 to ensure compatibility, replacing the original file.
    """
    temp_output = video_path.with_suffix(".temp.mp4")

    try:
        print(f"🔄 Re-encoding video: {video_path.name}")
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            str(temp_output)
        ], check=True)

        # Replace original file with re-encoded version
        temp_output.rename(video_path)
        print("✅ Video re-encoding complete!")

    except Exception as e:
        print(f"⚠️ Error re-encoding video: {e}")
        if temp_output.exists():
            temp_output.unlink()  # Remove broken file

def download_360_video(url, output_dir="downloads"):
    """
    Downloads a YouTube video in MP4 format and ensures it is always treated as a 360° video.
    After downloading, it re-encodes the video and injects 360° metadata.

    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the downloaded video
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Start with a placeholder; we'll override it after format selection
    ydl_opts = {
        "format": "best",
        "merge_output_format": "mp4",
        "outtmpl": f"{output_dir}/%(id)s.%(ext)s",
        "quiet": False,
        "no_warnings": False,
        "verbose": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first, don't download yet
            info = ydl.extract_info(url, download=False)

            # Filter formats to exclude stereoscopic and keep only MP4 video
            formats = info.get("formats", [])
            mono_formats = [
                f for f in formats
                if f.get("vcodec") != "none" and f.get("ext") == "mp4" and (
                    "stereo3d" not in f or f["stereo3d"] is None or f["stereo3d"].lower() == "none"
                )
            ]

            if not mono_formats:
                raise RuntimeError("❌ No suitable monoscopic MP4 format found.")

            # Choose highest resolution monoscopic format
            best_video_format = max(mono_formats, key=lambda f: f.get("height", 0))
            best_video_id = best_video_format["format_id"]
            print(f"🎯 Selected video format: {best_video_id} ({best_video_format.get('height', 'N/A')}p)")

            # Update ydl_opts with selected format
            ydl_opts["format"] = f"{best_video_id}+bestaudio[ext=m4a]/best[ext=mp4]"

        # Redo yt_dlp with updated format
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            video_path = Path(output_dir) / f"{info['id']}.mp4"
            metadata_path = Path(output_dir) / f"{info['id']}_metadata.json"

            print(f"\n✅ Download complete!")
            print(f"📂 Video saved in: {video_path}")
            print(f"📝 Metadata saved to: {metadata_path}")

            # Save basic metadata
            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "title": info.get("title", "Unknown"),
                        "uploader": info.get("uploader", "Unknown"),
                        "resolution": f"{info.get('width', 'Unknown')}x{info.get('height', 'Unknown')}",
                    },
                    f,
                    indent=4,
                )

            # Ensure compatibility by re-encoding
            reencode_video(video_path)

            # Inject 360° metadata
            inject_360_metadata(video_path)

    except Exception as e:
        print(f"❌ Error downloading video: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <YouTube_URL>")
        sys.exit(1)

    video_url = sys.argv[1]
    download_360_video(video_url)
