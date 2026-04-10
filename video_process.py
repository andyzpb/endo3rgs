import os
import cv2


def crop_lrtb(img, left, right, top, bottom):
    """Fixed crop: left/right/top/bottom are pixel integers."""
    h, w = img.shape[:2]
    x0 = int(max(0, left))
    x1 = int(min(w, w - right))
    y0 = int(max(0, top))
    y1 = int(min(h, h - bottom))
    if x1 <= x0 + 20 or y1 <= y0 + 20:
        return None
    return img[y0:y1, x0:x1]


def extract_cropped_frames_to_jpg(
    video_path: str,
    out_dir: str = "video_frames_cropped",
    crop_left: int = 763,
    crop_right: int = 233,
    crop_top: int = 78,
    crop_bottom: int = 78,
    start_frame: int = 0,
    end_frame: int | None = None,
    every_n: int = 1,             # 1=save every frame; 2=save every other frame
    jpg_quality: int = 95,        # 0~100
    write_timestamp_txt: bool = False,  # Also save frame timestamps (ms) to txt file
):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if end_frame is None:
        end_frame = total - 1 if total > 0 else 10**18  # Use large value when frame_count is not available

    if start_frame < 0:
        start_frame = 0

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # JPEG parameters
    jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(0, min(100, jpg_quality)))]

    ts_fp = None
    if write_timestamp_txt:
        ts_fp = open(os.path.join(out_dir, "timestamps_ms.txt"), "w", encoding="utf-8")

    saved = 0
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame0 = cap.read()
        if not ret:
            break  # Video finished reading or read failed

        # Sample frames by every_n
        if (frame_idx - start_frame) % every_n != 0:
            frame_idx += 1
            continue

        cropped = crop_lrtb(frame0, crop_left, crop_right, crop_top, crop_bottom)
        if cropped is None:
            print(f"[skip] frame {frame_idx}: too small after crop")
            frame_idx += 1
            continue

        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
        ok = cv2.imwrite(out_path, cropped, jpg_params)
        if not ok:
            print(f"[fail] write: {out_path}")
        else:
            saved += 1
            if ts_fp is not None:
                # Current frame timestamp (milliseconds)
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts_fp.write(f"{frame_idx}\t{t_ms:.3f}\n")

        if saved % 200 == 0:
            print(f"[info] saved {saved} frames... (current frame {frame_idx})")

        frame_idx += 1

    cap.release()
    if ts_fp is not None:
        ts_fp.close()

    print("\n===== Done =====")
    print(f"Video: {video_path}")
    print(f"FPS: {fps} | Total frames (reported): {total}")
    print(f"Crop (L,R,T,B): {crop_left}, {crop_right}, {crop_top}, {crop_bottom}")
    print(f"Saved frames: {saved}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    # ===== Crop parameters from existing calibration script =====
    base_left = 685
    base_right = 155
    add = 78

    CROP_LEFT = base_left + add   # 763
    CROP_RIGHT = base_right + add # 233
    CROP_TOP = add                # 78
    CROP_BOTTOM = add             # 78

    # ===== Update video path here =====
    VIDEO_PATH = "/home/mrlab/Documents/andrew/Robo_Ctrl/video_data/WIN_20260204_11_02_41_Pro.mp4"

    extract_cropped_frames_to_jpg(
        video_path=VIDEO_PATH,
        out_dir="video_frames_cropped_WIN_20260204_11_02_41_Pro",
        crop_left=CROP_LEFT,
        crop_right=CROP_RIGHT,
        crop_top=CROP_TOP,
        crop_bottom=CROP_BOTTOM,
        start_frame=0,
        end_frame=None,      # None=until end
        every_n=1,           # 1=save every frame
        jpg_quality=95,
        write_timestamp_txt=False,
    )
