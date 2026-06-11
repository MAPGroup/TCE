# TCE

Abstract
Reliable visual perception is fundamental to autonomous navigation and efficient maritime management. However, in complex marine environments, optical sensors face significant challenges. Environmental factors such as fog and low-light conditions inevitably lead to significant signal attenuation and noise. This degradation adversely impacts downstream perception algorithms, creating a critical bottleneck that ultimately jeopardizes navigation safety.
To mitigate the detrimental effects of image degradation, this study aims to capture the shared characteristics of multi-type degradations, enabling a unified parameter set to enhance images across diverse adverse conditions. Specifically, we decompose the objective of multi-scenario enhancement into two synergistic sub-tasks: texture reconstruction and color restoration, implemented via a dual-branch encoder-decoder architecture (TCE). Within this framework, an edge-information enhancement module—leveraging Large Kernel Differential Convolutional (LKD)—and a Color-Domain Reconstruction Branch (CRB) based on image decomposition are integrated to drive the restoration process. Furthermore, edge and reflectance images are incorporated as supervisory signals during training to assist the network in suppressing content-irrelevant interference. Experimental results on maritime datasets demonstrate that our approach effectively recovers fine-grained structural details concealed by fog or low-light, achieving superior visual quality and computational efficiency. Notably, the model maintains robust performance across different degradation types using a single set of weights. Object detection benchmarks further validate the practical utility of our method in bolstering downstream visual perception tasks under foggy and low-light maritime environments.

Dataset:
链接: https://pan.baidu.com/s/1kVYYMCReqzEDOTgOxMvbHg 提取码: syg4


Train:
python train.py --data_dir "train" --train_sets fog --test_sets fog --model_save_dir ".\output_result" -train_batch_size 16 -train_epoch 100

Test:
python test.py  --data_dir "test  --model_file ".\output_result\epoch100.pkl"  --model_save_dir ".\results"
