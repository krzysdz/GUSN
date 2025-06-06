import tkinter as tk
from typing import Callable

from PIL import Image, ImageDraw, ImageTk


class DrawingGUI:
    _app: tk.Tk | None = None
    _canvas: tk.Canvas | None = None
    _width_scale: tk.Scale | None = None
    _send_callback: Callable[[bytes], int] | None = None
    _send_btn: tk.Button | None = None
    _clr_btn: tk.Button | None = None
    _result_label: tk.Label | None = None
    result: int | None = None

    def __init__(self, cb: Callable[[bytes], int], scale=8):
        self._scale = scale
        self._image = Image.new("L", (28 * scale, 28 * scale))
        self._draw = ImageDraw.Draw(self._image)
        self._send_callback = cb

    def start(self):
        self._app = tk.Tk()
        self._app.title("Draw a digit to classify")

        self._canvas = tk.Canvas(
            self._app, width=28 * self._scale, height=28 * self._scale
        )
        self._canvas.grid(row=0, column=0, columnspan=2)

        self._canvas.bind("<B1-Motion>", lambda ev: self._handle_draw(ev))

        self._imgtk = ImageTk.PhotoImage(self._image)
        self._canvas.create_image(0, 0, anchor="nw", image=self._imgtk)

        width_label = tk.Label(self._app, text="Width")
        width_label.grid(row=1, column=0)
        self._width_scale = tk.Scale(
            self._app, orient="horizontal", from_=1.0, to=6 * self._scale
        )
        self._width_scale.set(int(1.5 * self._scale))
        self._width_scale.grid(row=1, column=1)

        self._send_btn = tk.Button(
            self._app, text="Send", command=lambda: self._send_image()
        )
        self._send_btn.grid(row=2, column=0)
        self._clr_btn = tk.Button(
            self._app, text="Clear", command=lambda: self._clear_image()
        )
        self._clr_btn.grid(row=2, column=1)

        self._result_label = tk.Label(self._app, text="Draw a digit and press Send")
        self._result_label.grid(row=3, column=0, columnspan=2, pady=3)

        self._app.mainloop()

    def _redraw_img(self):
        self._imgtk.paste(self._image)

    def _handle_draw(self, event: tk.Event):
        assert self._width_scale is not None
        half_w = int(self._width_scale.get() / 2.0)
        x, y = event.x, event.y
        self._draw.ellipse((x - half_w, y - half_w, x + half_w, y + half_w), 255)
        self._redraw_img()

    def _send_image(self):
        from fpga_predict import prepare_image

        img = prepare_image(self._image, scale=True)
        self.result = self._send_callback(img)
        self._result_label["text"] = f"Detected digit {self.result}"

    def _clear_image(self):
        self._image.paste(0, (0, 0, *self._image.size))
        self._redraw_img()


if __name__ == "__main__":

    def ex(img: bytes):
        print(img)
        return 5

    gui = DrawingGUI(ex)
    gui.start()
