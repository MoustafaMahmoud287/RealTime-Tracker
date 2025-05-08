# Real-Time Object Tracking (Graduation Project)

![Language](https://img.shields.io/badge/language-C++-blue.svg)
![CUDA](https://img.shields.io/badge/GPU-CUDA-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

![Demo](demo.gif) <!-- Replace with actual GIF file once uploaded -->

## Description
A real-time object tracking system developed using **C++** and **CUDA** to achieve high-performance tracking of objects in video frames. This system uses the **CIELUV color space** for robust color-based object detection under varying lighting conditions. It leverages **CUDA** to parallelize image processing tasks, ensuring real-time tracking even for high-resolution videos.

## Features
- 🎯 Real-time object tracking on video streams
- 🎨 Uses CIELUV color space for perceptual color accuracy
- 🧠 Custom thresholding and region-based analysis
- 🚀 CUDA acceleration for GPU-parallel image processing
- 🖼️ High performance on high-resolution video frames

## Demo
<img src="demo.gif" alt="Real-Time Tracking Demo" width="600" />

> To add your own demo: record your program in action, save as `demo.gif`, and place it in the repo root.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/MoustafaMahmoud287/RealTime-Tracker.git
cd RealTime-Tracker
