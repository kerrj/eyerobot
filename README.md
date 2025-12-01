# EyeRobot
This is the code release for the EyeRobot CoRL 2025 paper! 

It supports training and robot inference (only implemented for a UR5 setup), or simulated inference of the eyeball only (by running visual search policies on 360 videos).

# Dependencies

```bash
git clone --recurse-submodules git@github.com:kerrj/eyerobot.git
git submodule update --init --recursive
conda create -n eyeball python=3.10 -y
conda activate eyeball
cd eyeball
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
conda install -c conda-forge "ffmpeg>7"
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu126
pip install viser[examples]
```

## Hardware
CAD files for printing the eyeball are included inside `cad_files`. M3 and M2.5 screws are used for mounting motors to the printed parts. For the camera, we use [this](https://www.amazon.com/dp/B0C3C459JT?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1) 90fps 2.1mm lens camera, connected via USB. For the motors, we use the MS5005, 15-bit, RS485 hollow shaft direct drive servo modules from [LKMotor](http://en.lkmotor.cn/ProDetail.aspx?ProId=238), which can be purchased through AliExpress or Alibaba. To connect to these you will also need a USB to RS485 bridge which can be purchased from a variety of sources, we used [this link](https://www.amazon.com/dp/B081NBCJRS?ref=ppx_yo2ov_dt_b_fed_asin_title). For wiring the motors we recommend buying a pre-crimped ZH1.5 wire kit such as [this one](https://www.amazon.com/dp/B0C6WY7DZM), and any 24V DC power supply will suffice for powering the motors.

You can follow instructions from the manufacturer's website for downloading the motor tool (windows only) for calibrating the motor encoders. Inside the motor tool you can also set the ID for each motor for daisy-chaining them together. All code for interfacing with the motors via RS485 is in `lkmotor.py`. 

## Data
Data is hosted [here](https://drive.google.com/file/d/1v-yklVlefulhh6RwK6EJesS4-lbGukgf/view?usp=sharing). Download it as `./data` in this repository.

To visualize a 360 video, you can run

```bash
python scripts/test360.py --video video.mp4
```

which launches a visualizer of a video.

### Data structure
The robot data looks like this:
```
data/
  demos/
    dir/
      processed/
      sequences/
        downsampled_<timestamp>_joint_data.h5
        downsampled.mp4 # low res video
        trimmed.mp4 # high res trimmed
        wrist_trimmed.mp4 # wrist video (if available)
        zed_trimmed.mp4 # 3rd person video (if available)
```

Data loading is done inside gyms.py, and is used in the train_bcrl.py script.

## Training

To run pretraining, run `torchrun --nproc_per_node <ngpus> scripts/train_avp.py` which will train the eyeball to look around inside the robot data provided. It trains an eyeball policy on all the available data, conditioning on the provided target prompts inside the file. For robot pretraining it samples one still frame of the video to train on  each episode.

You can test this script on a custom video with `scripts/test_visual_search_sim.py`

Training on custom videos requires placing your 360 videos in a similar format as the robot data, and providing positive crop "prompts" for it. Alternatively with some code modifications you could run on raw videos themselves.

To run BCRL (eye+robot) training, run `torchrun --nproc_per_node <gpu> scripts/train_bcrl.py`. This will launch BC-RL pretraining on all the tasks at once (you can modify this by changing the specified data paths). To load from a pretrained checkpoint (i.e from train_avp.py), specify the flag `--load-eye-weights-from <checkpoint.pt>`.

# Release TODOs
- [ ] Write instructions for collecting your own 360 data (UR5 specific)
