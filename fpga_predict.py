from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps
from serial import Serial, PARITY_EVEN

BAUD = 921600
PARITY = PARITY_EVEN
_serial: Serial | None = None


def get_serial(port: str):
    global _serial

    if _serial is None or _serial.closed:
        _serial = Serial(port, BAUD, parity=PARITY)
    return _serial


def prepare_image(image: Path | Image.Image, invert=False, scale=False):
    if isinstance(image, Path):
        image = Image.open(image)

    if invert:
        image = ImageOps.invert(image)

    if image.size != (28, 28):
        if not scale:
            raise RuntimeError(
                f"Image has dimensions other than 28x28 ({image.size}), but scaling is not enabled."
            )
        image = image.resize((28, 28), resample=Image.Resampling.LANCZOS)

    return image.convert("L").tobytes()


def process_image(serial: Serial, image: Path | Image.Image, invert=False, scale=False):
    image_bytes = prepare_image(image, invert, scale)
    return process_raw(serial, image_bytes)


def process_raw(serial: Serial, data: bytes):
    assert len(data) == 28 * 28
    serial.write(data)
    response = serial.read()
    # The response is ASCII
    return response[0] - 0x30


def main(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Send image to FPGA for digit recognition and read the result"
    )
    parser.add_argument(
        "serial", help="serial port e.g. /dev/ttyUSB0 (Linux) or COM1 (Windows)"
    )
    parser.add_argument("-g", "--gui", help="Use GUI for drawing", action="store_true")
    cli_group = parser.add_argument_group(
        "CLI arguments", "Arguments for use in CLI mode (with no --gui switch)"
    )
    cli_group.add_argument(
        "-i",
        "--invert",
        help="invert image colours (use for black on white), applies to all images",
        action="store_true",
    )
    cli_group.add_argument(
        "-s",
        "--scale",
        help="allow scaling images (Lanczos) to 28x28 pixels input format",
        action="store_true",
    )
    cli_group.add_argument(
        "image",
        help="image(s) to process, required if not in GUI mode",
        type=Path,
        nargs="?",
    )

    args = parser.parse_args(argv)
    serial = get_serial(args.serial)
    gui: bool = args.gui

    if gui:
        from drawing_gui import DrawingGUI

        ui = DrawingGUI(lambda data: process_raw(serial, data))
        ui.start()
    else:
        invert: bool = args.invert
        scale: bool = args.scale
        files: list[Path] | None = args.image

        if files is None:
            parser.error("image argument is required in CLI mode")

        for img in files:
            print(f"Sending file {img}...", end="")
            detected = process_image(serial, img, invert, scale)
        print(f" detected {detected}")


if __name__ == "__main__":
    main()
