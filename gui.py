"""Desktop GUI for handwritten digit recognition: draw or upload a digit."""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw
from digit_model import get_model, preprocess


model = get_model()


def predict_array(arr: np.ndarray) -> tuple[int, float]:
    probs = model.predict(arr, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence


class App(tk.Tk):
    CANVAS_SIZE = 280          # 10× MNIST size for comfortable drawing
    PEN_WIDTH   = 18

    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.resizable(False, False)
        self._build_ui()
        self._reset_canvas()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=8, pady=6)

        # Left: drawing canvas
        left = tk.Frame(self)
        left.grid(row=0, column=0, **pad)

        tk.Label(left, text="Draw a digit (0–9):", font=("Helvetica", 12)).pack(anchor="w")

        self.canvas = tk.Canvas(
            left,
            width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
            bg="white", cursor="pencil",
            relief="solid", borderwidth=1,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_draw)
        self.canvas.bind("<Motion>",          self._on_draw)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Leave>",           self._on_release)

        btn_row = tk.Frame(left)
        btn_row.pack(fill="x", pady=(6, 0))
        tk.Button(btn_row, text="Clear",    width=10, command=self._reset_canvas).pack(side="left")
        tk.Button(btn_row, text="Predict",  width=10, command=self._predict_canvas, bg="#4CAF50", fg="white").pack(side="left", padx=4)
        tk.Button(btn_row, text="Upload…",  width=10, command=self._upload_image).pack(side="right")

        # Right: result panel
        right = tk.Frame(self, relief="groove", borderwidth=2)
        right.grid(row=0, column=1, sticky="ns", **pad)

        tk.Label(right, text="Prediction", font=("Helvetica", 13, "bold")).pack(pady=(10, 0))

        self.digit_label = tk.Label(right, text="—", font=("Helvetica", 72, "bold"), fg="#1565C0", width=3)
        self.digit_label.pack()

        self.conf_label = tk.Label(right, text="Confidence: —", font=("Helvetica", 11))
        self.conf_label.pack()

        # Bar chart of all probabilities
        tk.Label(right, text="All probabilities:", font=("Helvetica", 10)).pack(pady=(12, 2))
        self.bars = []
        self.bar_labels = []
        bar_frame = tk.Frame(right)
        bar_frame.pack(padx=10, pady=(0, 10))

        for i in range(10):
            row_f = tk.Frame(bar_frame)
            row_f.pack(fill="x", pady=1)
            tk.Label(row_f, text=str(i), width=2, font=("Courier", 10)).pack(side="left")
            bar_bg = tk.Frame(row_f, bg="#e0e0e0", width=140, height=14, relief="flat")
            bar_bg.pack(side="left", padx=2)
            bar_bg.pack_propagate(False)
            bar = tk.Frame(bar_bg, bg="#1565C0", width=0, height=14)
            bar.place(x=0, y=0, relheight=1.0)
            pct_lbl = tk.Label(row_f, text="  0%", width=5, font=("Courier", 9))
            pct_lbl.pack(side="left")
            self.bars.append(bar)
            self.bar_labels.append(pct_lbl)

    # ── Canvas helpers ─────────────────────────────────────────────────────────
    def _reset_canvas(self):
        self.canvas.delete("all")
        self._pil_image  = Image.new("RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), "white")
        self._pil_draw   = ImageDraw.Draw(self._pil_image)
        self._last_xy    = None
        self._is_drawing = False
        self.digit_label.config(text="—")
        self.conf_label.config(text="Confidence: —")
        for bar, lbl in zip(self.bars, self.bar_labels):
            bar.place(x=0, y=0, width=0, relheight=1.0)
            lbl.config(text="  0%")

    def _on_press(self, event):
        self._is_drawing = True
        x, y = event.x, event.y
        r = self.PEN_WIDTH // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self._pil_draw.ellipse([x-r, y-r, x+r, y+r], fill="black")
        self._last_xy = (x, y)

    def _on_release(self, event):
        self._is_drawing = False
        self._last_xy = None

    def _on_draw(self, event):
        if not self._is_drawing:
            return
        x, y = event.x, event.y
        r = self.PEN_WIDTH // 2
        # Draw a filled oval at the current point
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self._pil_draw.ellipse([x-r, y-r, x+r, y+r], fill="black")
        # Connect to the previous point with a thick line to avoid gaps
        if self._last_xy:
            lx, ly = self._last_xy
            self.canvas.create_line(lx, ly, x, y, fill="black", width=self.PEN_WIDTH, capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self._pil_draw.line([lx, ly, x, y], fill="black", width=self.PEN_WIDTH)
        self._last_xy = (x, y)

    # ── Prediction helpers ─────────────────────────────────────────────────────
    def _predict_canvas(self):
        arr = preprocess(self._pil_image)
        self._show_result(arr)

    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select digit image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            arr = preprocess(img)
            self._show_result(arr)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _show_result(self, arr: np.ndarray):
        probs = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(probs))
        conf  = float(probs[digit])

        self.digit_label.config(text=str(digit))
        self.conf_label.config(text=f"Confidence: {conf*100:.1f}%")

        for i, (bar, lbl) in enumerate(zip(self.bars, self.bar_labels)):
            pct = probs[i] * 100
            w   = int(pct / 100 * 140)
            color = "#1565C0" if i == digit else "#90CAF9"
            bar.place(x=0, y=0, width=w, relheight=1.0)
            bar.config(bg=color)
            lbl.config(text=f"{pct:4.1f}%")


if __name__ == "__main__":
    App().mainloop()
