# Sherd Align Programme

This programme is designed to align sherds in RGB images to respective depth image.

---

### How To Use:
1) Download the "align.py" file and move it to the directory of your choosing.
2) Run with `python align.py -d <path-to-rgb-directory> <path-to-depth-directory>`
3) *Optional*: The default output path is `./output`. To change this, use the option `-o <path-to-store-results>`
>The output directory will be created if it does not exist.

### Dependencies
- OpenCV (4.5.1)
- Python (3.9)
- Numpy (1.22.1)