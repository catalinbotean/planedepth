"""Render a snippet of HTML to a tightly cropped PNG of a given height.

ffmpeg here is built without freetype, so text comes from QuickLook's HTML
renderer instead. The output is auto-cropped to the ink, which removes any need
to know how QuickLook scaled the page.
"""
import os, subprocess, sys, tempfile

def render(html, out_png, target_h, size=3030):
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "t.html")
    open(src, "w").write(
        "<html><body style='margin:0;padding:0;background:#fff'>" + html + "</body></html>")
    subprocess.run(["qlmanage", "-t", "-s", str(size), "-o", tmp, src],
                   capture_output=True)
    raw_png = os.path.join(tmp, "t.html.png")
    if not os.path.isfile(raw_png):
        raise SystemExit("qlmanage produced nothing for " + out_png)

    w = h = size
    gray = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", raw_png,
                           "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                          capture_output=True).stdout
    x0, y0, x1, y1 = w, h, -1, -1
    for y in range(h):
        row = gray[y * w:(y + 1) * w]
        if min(row) > 245:
            continue
        y0 = min(y0, y); y1 = max(y1, y)
        for x in range(w):
            if row[x] <= 245:
                x0 = min(x0, x); x1 = max(x1, x)
    if x1 < 0:
        raise SystemExit("nothing rendered for " + out_png)

    pad = 2
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    cw = min(w - x0, x1 - x0 + 1 + 2 * pad); ch = min(h - y0, y1 - y0 + 1 + 2 * pad)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw_png, "-vf",
                    "crop={}:{}:{}:{},scale=-1:{}".format(cw, ch, x0, y0, target_h),
                    out_png], check=True)

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], int(sys.argv[3]))
