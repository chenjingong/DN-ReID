# Day-Night Cross-domain Vehicle Re-identification
Authors:Hongchao Li, Jingong Chen, Aihua Zheng, Yong Wu*, Yonglong Luo1
Paper:CVPR2024

### Abstract
 Previous advances in vehicle re-identification (ReID) are mostly reported under favorable lighting conditions, while cross-day-and-night performance is neglected, which greatly hinders the development of related traffic intelligence applications. This work instead develops a novel Day-Night Dual-domain Modulation (DNDM) vehicle reidentification framework for day-night cross-domain traffic scenarios. Specifically, a unique night-domain glare suppression module is provided to attenuate the headlight glare from raw nighttime vehicle images. To enhance vehicle features under low-light environments, we propose a dual-domain structure enhancement module in the feature extractor, which enhances geometric structures between appearance features. To alleviate day-night domain discrepancies, we develop a cross-domain class awareness module that facilitates the interaction between appearance and structure features in both domains. In this work, we address the Day-Night cross-domain ReID (DN-ReID) problem and provide a new cross-domain dataset named DNWild, including day and night images of 2,286 identities, giving in total 85,945 daytime images and 54,952 nighttime images. Furthermore, we also take into account the matter of balance between day and night samples, and provide a dataset called DN-348. Exhaustive experiments demonstrate the robustness of the proposed framework in the DNReID problem. The code and benchmark are released at https://github.com/chenjingong/DN-ReID.

 ### Dataset download
 We put the path in the data_path/data_path.txt

 ### Training

   Train a model by
  ```bash
python train.py --dataset dn348 --lr 0.1 --method agw --gpu 1
```

  - `--dataset`: which dataset "dn348" or "dnwild".

  - `--lr`: initial learning rate.
  
  -  `--method`: method to run or baseline.
  
  - `--gpu`:  which gpu to run.

You may need mannully define the data path first.

 ### Test

 Test a model on dn348 or dnwild dataset by 
  ```bash
python test.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
  - `--dataset`: which dataset "dn348" or "dnwild".
  
  - `--resume`: the saved model path.
  
  - `--gpu`:  which gpu to run.

### Citation
If you use the dataset, please cite the following paper:
```
@inproceedings{Li_2024_CVPR,
  title={Day-Night Cross-Domain Vehicle Re-Identification},
  author={Li, Hongchao and Chen, Jingong and Zheng, Aihua and Wu, Yong and Luo, YongLong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```