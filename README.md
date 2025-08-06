# DH-MSVM-Datasets-and-Benchmarks
DH-MSVM Datasets and Benchmarks

This repository contains datasets and reference implementations for the paper:  
**"DH-MSVM: A Hybrid Algorithm for Seeking Quality Support Vectors in Distributed Learning"**  

## 9 real-world datasets
| Dataset    | \(D_{tr}\) | \(D_{te}\) | Dimension |
|------------|------------|------------|-----------|
| Skin       | 196045     | 49012      | 3         |
| Diabetes   | 46800      | 30000      | 8         |
| Heart      | 17000      | 10000      | 13        |
| German     | 70000      | 30000      | 20        |
| Ijcnn      | 127787     | 63894      | 22        |
| Acoustic   | 73896      | 24632      | 50        |
| Splice     | 43500      | 20000      | 60        |
| Isbi4      | 360000     | 135600     | 1681      |
| TV_News    | 86457      | 43228      | 4125      |

⚠ **Note**: For datasets exceeding 25MB, please download from [Dataset Portal](https://pan.baidu.com/s/1wRHuLsHMpnZjhpA-AX7isQ?pwd=wieh code: wieh). 

## Implemented Baselines
| Algorithm       | Reference                          |
|-----------------|------------------------------------|
| LGSVM | Marchetti, F., Perracchione, E., 2022. Local-to-global support vector machines (lgsvms). Pattern Recognition 132, 108920.|
| DMSVM           | Zou, B., Jiang, H., Xu, C., Xu, J., You, X., Tang, Y.Y., 2023. Learning performance of weighted distributed learning with support vector machines. IEEE Transactions on Cybernetics 53, 4630–4641.|
| DRSVOR       | Liu, H., Tu, J., Gao, A., Li, C., 2024. Distributed robust support vector ordinal regression under label noise. Neurocomputing 598, 128057.|
| DESVM       | Xu, W., Liu, J., Lian, H., 2024. Distributed estimation of support vector machines for matrix data. IEEE transactions on neural networks an learning systems. |
| DFSVM       | Li, W., Shan, W., Liu, M., 2025. A distributed algorithm for fuzzy support vector machine on multi-source data-driven credit scoring. Engineering Applications of Artiffcial Intelligence 143,110009.|
| TDLDA |Li, M., Zhao, J., 2022. Communication-efffcient distributed linear discriminant analysis for binary classiffcation. Statistica Sinica 32, 1343–1361.|
| RSLDA        |Wang, J., Wang, H., Nie, F., Li, X., 2022a. Ratio sum versus sum ratio for linear discriminant analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 10171–10185. |
| DDNM        | Luo, X., Wen, X., Zhou, M., Abusorrah, A., Huang, L., 2021. Decisiontree-initialized dendritic neuron model for fast and accurate data classiffcation. IEEE Transactions on Neural Networks and Learning Systems 33, 4173–4183. |
| SNN10       | Klambauer, G., Unterthiner, T., Mayr, A., Hochreiter, S., 2017. Self-normalizing neural networks. Advances in neural information processing systems 30.|

⚠ **Note**: Some compared algorithms did not release original complete implementations in their papers. This repository only provides reproduced core code snippets for reference. For full implementations, please contact the respective authors.
