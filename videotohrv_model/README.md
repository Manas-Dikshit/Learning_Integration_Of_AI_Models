# HeartRateVariability-rPPG

Welcome to the HeartRateVariability-rPPG repository! This machine learning model is designed to process video inputs to estimate Heart Rate Variability (HRV) metrics using the remote Photoplethysmography (rPPG) method. This innovative approach allows for non-contact measurement of heart rate variability by analyzing the subtle changes in skin color that occur with each heartbeat. Our model leverages the power of advanced signal processing and computer vision techniques to provide accurate HRV metrics, which are crucial for assessing stress, cardiovascular health, and overall well-being.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Libraries and Resources](#libraries-and-resources)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Description

This project utilizes state-of-the-art algorithms and methodologies derived from key resources and libraries such as yarppg, HeartPy, OpenCV, and NeuroKit2 to extract, process, and analyze the video data for HRV metrics. The rPPG method applied here offers a unique advantage in remote health monitoring, fitness tracking, and psychological research by providing a non-invasive means of measuring heart rate variability through video analysis.


## Usage



## Libraries and Resources

This project makes extensive use of the following libraries and resources:

- **[yarppg](https://github.com/SamProell/yarppg):** For insights and methodologies on implementing the rPPG technique.
- **[HeartPy](https://github.com/paulvangentcom/heartrate_analysis_python):** Used for heart rate signal processing and HRV metrics analysis.
- **[OpenCV](https://opencv.org/):** For video processing and computer vision operations.
- **[NeuroKit2](https://neurokit2.readthedocs.io/en/latest/):** An advanced tool for signal processing, including HRV metrics.

## How It Works

The model processes the input video to detect the subject's face and, specifically, the cheek region, where skin color changes are more pronounced and less affected by motion artifacts. It then applies signal processing techniques to extract the rPPG signal from the color variations over time. This signal is further processed to compute HRV metrics, which include but are not limited to SDNN, RMSSD, and frequency domain metrics.

## Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

This project would not have been possible without the foundational work and insights provided by the following resources and libraries:

- The developers and contributors of **yarppg**, **HeartPy**, **OpenCV**, and **NeuroKit2** for their invaluable libraries and tools that made this project feasible.
- The academic and research community for advancing the field of non-contact HRV measurement through rPPG and related technologies.

Please note that this project is for educational and research purposes only and may not be suitable for clinical use.