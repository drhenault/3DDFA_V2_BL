# Towards Fast, Accurate and Stable 3D Dense Face Alignment

[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/babblelabs/3DDFA_V2.svg)

A fork of 3DDFA_V2, modified for internal purposes.

## Installation (Linux)

1. **Clone the repository (with submodules):**
   ```bash
   git clone --recursive git@github.com:babblelabs/3ddfa_v2
   cd 3ddfa_v2
   ```

2. **Import FaceNet checkpoint(s).**  
   Get pre-trained model checkpoints from the [FaceNet GitHub repository](https://github.com/davidsandberg/facenet) (see its README for download links). Create a subdirectory for each checkpoint under `facenet/models/` and place the checkpoint files there (e.g. `facenet/models/20180402-114759/`).

3. **TX Audio Pipeline**  
   Obtain a copy of the TX audio pipeline repository. Run its demo TX pipeline script with input paths pointing at this repo’s `./input` directory. 

4. **Run the setup script** to create and populate the virtual environments:
   ```bash
   ./setup_script.sh
   ```
   This sets up the FaceNet and 3DDFA_v2 virtual environments and runs the Cython build.

## Processing one video and rendering

1. Put your video file in `./inputs` and enrollment avatars for face matching in `./enrollment-avatars`.
2. In `stitch_demo.sh`, set the correct path to the TX audio pipeline repository (the `AUDIO_PIPELINE_DIRECTORY_PATH` variable).
3. Run `./stitch_demo.sh` and wait for the pipeline to finish. Output will appear in `./outputs`.

---

## Legacy script usage

### Video analysis
Performs an analysis of a video file. Collects information about headcount, head position, orientation, and proximity.

Syntax:

`$ (venv) python3 demo_webcam_smooth.py [options]`

Currently working options are:

`-f`, `--video_fp`: path to the input video file.

`--dump_results`: can be `true` or `false`. Controls whether results are dumped to a file. If `true`, saves the results in comma-separated value format in `dumps/` directory.

`-c`, `--config`: config to use by 3DDFA network. Defines used checkpoint file, architecture, number of parameters and a few other variables. Defaults to `configs/mb1_120x120.yml`.

<!-- TODO: add description of the CSV file format -->

Upstream repository:
[3DDFA_v2]
by [Jianzhu Guo](https://guojianzhu.com), [Xiangyu Zhu](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/), [Yang Yang](http://www.cbsr.ia.ac.cn/users/yyang/main.htm), Fan Yang, [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/) and [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ).
The upstream code repo is owned and maintained by [Jianzhu Guo](https://guojianzhu.com).