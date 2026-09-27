# heartvis

`heartvis` renders a 720p60 music visualizer from a stereo audio file with an embedded CUE sheet.

Each cue track is paired with shuffled artwork. The artwork is scaled to cover the frame and gently panned across the track duration. A stereo FFT is drawn as a luminous heart: the left channel forms the left half, the right channel forms the right half, frequency rises from the bottom point toward the cleft, and magnitude pushes the curve outward.

FFmpeg handles audio decoding and final encoding. Python, NumPy, and Pillow generate the video frames.

## Requirements

- Python 3
- FFmpeg and ffprobe
- NumPy
- Pillow
- An FFmpeg build with `libx264` and FLAC support

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Usage

The input needs an embedded CUE sheet with `TRACK`, `TITLE`, `PERFORMER`, and `INDEX 01` entries.

```bash
python heartvis.py mix.flac \
    --images ./images \
    --seed 69420 \
    -o mix.mkv
```

By default, `heartvis` renders 30 seconds. Use `--duration` for a longer render and `--start` to begin at an offset.

```bash
python heartvis.py mix.flac \
    --images ./images \
    --start 120 \
    --duration 30 \
    --seed 69420 \
    -o preview.mkv
```

For a full-length render, set `--duration` to a value at least as long as the remaining audio. Requests beyond EOF are automatically clamped.

## Artwork

`--images` is required. Supported formats are JPEG, PNG, WebP, BMP, and TIFF.

Image filenames are sorted and then shuffled. One image is assigned to each cue track. If there are fewer images than tracks, the pool is reshuffled and reused. Extra images are ignored.

Portrait images pan vertically; wide images pan horizontally. The first and last 5% of each cue are held still, with smooth motion through the middle 90%.

## Track text

Artist and title are shown at the lower left at the beginning of each cue. The default timing is a 1.5 second fade in, 8 seconds fully visible, and a 1.5 second fade out.

The default font path is `/usr/share/fonts/office/bierstadt-d.ttf`. Override it on systems where that font is unavailable:

```bash
python heartvis.py mix.flac \
    --images ./images \
    --font /path/to/font.ttf \
    -o mix.mkv
```

Timing can be changed with `--text-hold` and `--text-fade`.

## Deterministic renders

`--seed` controls both artwork assignment and the independent hue-cycle rates of the left and right spectrum curves. Reusing the same seed with the same image set reproduces those choices.

If no seed is supplied, a random one is generated and printed at startup so the render can be reproduced later.

## Output

The output is an MKV containing:

- 1280×720 H.264 video at 60 fps
- lossless 24-bit FLAC audio

The audio is decoded independently by FFmpeg for analysis and for the final output encode.
