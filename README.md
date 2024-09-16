# A Dual-Level Cancelable Framework for Palmprint Verification and Hack-Proof Data Storage
This repository is a PyTorch implementation of DCPV (accepted by IEEE Transactions on Information Forensics and Security).

#### Abstract
In recent years, palmprints have been extensively utilized for individual verification. The abundance of sensitive information in palmprint data necessitates robust protection to ensure security and privacy without compromising system performance. Existing systems frequently use cancelable transformations to protect palmprint templates. However, if an adversary gains access to the stored database, they could initiate a replay attack before the system detects the breach and can revoke and replace the reference template. To address replay attacks while meeting template protection criteria, we propose a dual-level cancelable palmprint verification framework. In this framework, the reference template is initially transformed using a cancelable competition hashing network with a first-level token, enabling the end-to-end generation of cancelable templates. During enrollment, the system creates a negative database (NDB) using a second-level token for further protection. Due to the unique NDB-to-vector matching characteristic, a replay attack involving the matching between the reference template and a compromised instance in NDB form is infeasible. This approach effectively addresses the replay attack problem at its root. Furthermore, the dual-level protected reference template enjoys heightened security, as reversing the NDB is NP-hard. We also propose a novel NDB-to-vector matching algorithm based on matrix operations to expedite the matching process, addressing the inefficiencies of previous NDB methods reliant on dictionary-based matching rules. Extensive experiments conducted on public palmprint datasets confirm the effectiveness and generality of the proposed framework.

#### Citation
If our work is valuable to you, please cite our work:
```
@ARTICLE{yang2023ccnet,
  author={Yang, Ziyuan and Kang, Ming and Teoh, Andrew Beng Jin and Gao, Chengrui and Chen, Wen and Zhang, Bob and Zhang, Yi},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Comprehensive Competition Mechanism in Palmprint Recognition}, 
  year={2024},
  doi={110.1109/TIFS.2024.3461869}
```

#### Requirements

If you have already tried our previous works [CCNet](https://github.com/Zi-YuanYang/CCNet/) and [CO3Net](https://github.com/Zi-YuanYang/CO3Net), you can skip this step.

Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Training
We set the recognition network as CCNet as default, you can change it to other networks. After you prepare the training and testing texts, then you can directly run our training code as follows:

```
python train.py --id_num xxxx --train_set_file xxxx --test_set_file xxxx --des_path xxxx --path_rst xxxx
```

* batch_size: the size of batch to be used for local training. (default: ```1024```)
* epoch_num: the number of total training epoches. (default: ```3000```)
* temp: the value of the tempture in our contrastive loss. (default: ```0.07```)
* weight1: the weight of cross-entropy loss. (default: ```0.8```)
* weight2: the weight of contrastive loss. (default: ```0.2```)
* com_weight: the weight of the traditional competition mechanism. (default: ```0.8```)
* id_num: the number of ids in the dataset.
* gpu_id: the id of training gpu.
* lr: the inital learning rate. (default: ```0.001```)
* redstep: the step size of learning scheduler. (default: ```500```)
* test_interval: the interval of testing.
* save_interval: the interval of saving.
* train_set_file: the path of training text file.
* test_set_file: the path of testing text file.
* des_path: the path of saving checkpoints.
* path_rst: the path of saving results.


#### Vlidation
Our negative database is only utilized in the reference stage, so please validate the performance as follows:
```
python Test_NDB.py --train_set_file xxxx --test_set_file xxxx --id_num xxx --path_rst xxx --batch_size xxxx --model_path xxx
```

#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.
