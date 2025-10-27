import tkinter as tk
from tkinter import filedialog
import cv2
import pandas as pd
import base64
import threading
import numpy as np

class VideoComparator:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Control Window")
        self.root.geometry("300x100")
        tk.Label(root, text="Focus this window for keyboard control (e/r/j/k, left/right click)").pack()

        self.csv_path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV Files", "*.csv")])
        if not self.csv_path:
            print("No file selected.")
            exit(0)

        df = pd.read_csv(self.csv_path, header=None)
        self.video_pairs = [(base64.b64decode(a).decode(), base64.b64decode(b).decode()) for a, b in df.values]

        self.current_index = 0
        self.speed = 1.0
        self.running = True
        self.restart_flag = threading.Event()

        # 键盘绑定到Tk窗口
        root.bind("<Key>", self.key_handler)
        root.bind("<Button-1>", lambda e: self.seek(-10))
        root.bind("<Button-3>", lambda e: self.seek(10))

        # 播放线程
        self.thread = threading.Thread(target=self.play_loop, daemon=True)
        self.thread.start()

    def play_loop(self):
        while self.running:
            self.restart_flag.clear()
            left_path, right_path = self.video_pairs[self.current_index]
            cap1 = cv2.VideoCapture(left_path)
            cap2 = cv2.VideoCapture(right_path)
            fps = cap1.get(cv2.CAP_PROP_FPS)
            total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
            total_seconds = total_frames / fps if fps > 0 else 0

            while cap1.isOpened() and cap2.isOpened() and self.running:
                if self.restart_flag.is_set():
                    break

                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break

                combined = cv2.hconcat([frame1, frame2])

                # 信息栏：黑底白字，高100px
                h, w, _ = combined.shape
                overlay = combined.copy()
                cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, combined, 0.4, 0, combined)

                pos_frames = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))
                seconds = pos_frames / fps if fps > 0 else 0
                info = f"Speed: {self.speed:.2f}x | Group: {self.current_index+1}/{len(self.video_pairs)} | Time: {seconds:.1f}/{total_seconds:.1f}s"
                cv2.putText(combined, info, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("Video Comparator", combined)
                key = cv2.waitKey(int((1000 / fps) / self.speed))
                if key == 27:  # ESC退出
                    self.running = False
                    break

            cap1.release()
            cap2.release()

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
            if self.current_index < len(self.video_pairs) - 1:
                self.current_index += 1
                self.restart_flag.set()
        print(f"Speed: {self.speed:.2f}x | Group: {self.current_index+1}")

    def seek(self, seconds):
        print(f"Seek {seconds:+d}s (not implemented for OpenCV streams)")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoComparator(root)
    root.mainloop()
