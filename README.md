# Payload Processor
Source code for the paper "Payload Processor: Message authentication for in-vehicle CAN bus using data compression and tag filling".

By releasing our source code, we aim to facilitate research in this area and encourage collaboration and innovation in the scientific community. We hope that our contribution will help advance the state of the art and inspire others to build on our work.


## 1.Source code
Whole Source file are implemented in Python language.

To run this code successfully, do:

(1) Download public available datasets according to the links provided in the paper and put them under the folder "Payload-Processor\dataset\". 

(2) Format the dataset using "dataset_format_transform.py" files.

(3) Uncomment the function in the "main.py" file and run the file.

(4) If you have two raspberry pi development boards and equipped with CAN communication environment, run the "ecu1_send.py" and "ecu2_receive.py" files respectively.


## 2.Paper details
If you are interested in our work, please access following URL:

https://doi.org/10.1016/j.comnet.2025.111061


If you use our resources, please cite our work:

MLA:
```
Zhang, Guiqi, et al. "Payload Processor: Message authentication for in-vehicle CAN bus using data compression and tag filling." Computer Networks (2025): 111061.
```


BibText:
```
@article{zhang2025payload,
  title={Payload Processor: Message authentication for in-vehicle CAN bus using data compression and tag filling},
  author={Zhang, Guiqi and Shen, Jun and Li, Jiangtao and Qin, Wutao and Li, Yufeng},
  journal={Computer Networks},
  pages={111061},
  year={2025},
  publisher={Elsevier}
}
```
