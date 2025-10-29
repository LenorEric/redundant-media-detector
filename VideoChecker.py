import tkinter as tk
from tkinter import filedialog
import cv2
import csv
import base64
import threading
import numpy as np
import math


class MultiVideoComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Control")
        self.root.geometry("360x120")
        tk.Label(root, text="Focus here for control (e/r/j/k, mouse left/right)").pack()

        # 选择 CSV
        self.csv_path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV Files", "*.csv")])
        if not self.csv_path:
            print("No file selected.")
            exit(0)

        # 使用csv.reader避免列数不一致报错
        self.video_groups = []
        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                paths = []
                for cell in row:
                    cell = cell.strip()
                    if cell:
                        try:
                            paths.append(base64.b64decode(cell).decode())
                        except Exception:
                            print(f"Warning: Invalid base64 -> {cell[:20]}...")
                if paths:
                    self.video_groups.append(paths)

        self.max_videos = 9
        if any(len(g) > self.max_videos for g in self.video_groups):
            print(f"Warning: Some groups exceed {self.max_videos} videos, truncated to 9.")

        self.current_index = 0
        self.speed = 1.0
        self.running = True
        self.restart_flag = threading.Event()

        root.bind("<Key>", self.key_handler)
        root.bind("<Button-1>", lambda e: self.seek(-10))
        root.bind("<Button-3>", lambda e: self.seek(10))

        self.thread = threading.Thread(target=self.play_loop, daemon=True)
        self.thread.start()

    def play_loop(self):
        while self.running:
            self.restart_flag.clear()
            paths = self.video_groups[self.current_index][:self.max_videos]
            caps = [cv2.VideoCapture(p) for p in paths]
            fps_list = [cap.get(cv2.CAP_PROP_FPS) or 30 for cap in caps]
            fps = min(fps_list)
            total_frames = int(min([cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps]))
            total_seconds = total_frames / fps if fps > 0 else 0

            while all(cap.isOpened() for cap in caps) and self.running:
                if self.restart_flag.is_set():
                    break

                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    frames.append(frame)

                n = len(frames)
                grid_cols = math.ceil(math.sqrt(n))
                grid_rows = math.ceil(n / grid_cols)
                max_w, max_h = (2560 * 2 // 3 - 200) // grid_cols, (1440 * 3 // 5 - 200) // grid_rows

                resized = [cv2.resize(f, (max_w, max_h)) for f in frames]
                while len(resized) < grid_rows * grid_cols:
                    resized.append(np.zeros_like(resized[0]))
                rows = [cv2.hconcat(resized[i * grid_cols:(i + 1) * grid_cols]) for i in range(grid_rows)]
                combined = cv2.vconcat(rows)

                pos_frames = int(caps[0].get(cv2.CAP_PROP_POS_FRAMES))
                seconds = pos_frames / fps if fps > 0 else 0

                # 顶栏 30px 黑底白字
                h, w, _ = combined.shape
                overlay = combined.copy()
                cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, combined, 0.3, 0, combined)

                info = f"Speed: {self.speed:.2f}x | Group: {self.current_index + 1}/{len(self.video_groups)} | Time: {seconds:.1f}/{total_seconds:.1f}s"
                cv2.putText(combined, info, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("Multi Video Comparator", combined)
                key = cv2.waitKey(int((1000 / fps) / self.speed))
                if key == 27:
                    self.running = False
                    break

            for cap in caps:
                cap.release()

        cv2.destroyAllWindows()

    def key_handler(self, event):
        k = event.keysym.lower()
        if k == 'j':
            self.speed /= 1.5
        elif k == 'k':
            self.speed *= 1.5
        elif k == 'e':
            if self.current_index > 0:
                self.current_index -= 1
                self.restart_flag.set()
        elif k == 'r':
            if self.current_index < len(self.video_groups) - 1:
                self.current_index += 1
                self.restart_flag.set()
        print(f"Speed: {self.speed:.2f}x | Group: {self.current_index + 1}")

    def seek(self, seconds):
        print(f"Seek {seconds:+d}s (not implemented for OpenCV streams)")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiVideoComparator(root)
    root.mainloop()
