# Traffic Forecasting on New Roads Using Spatial Contrastive Pre-Training (SCPT)

![Visual_abstract](/vizabs2.png) Our novel traffic forecasting framework, Spatial Contrastive Pre-Training (SCPT), enables accurate forecasts on new roads (orange) that were not seen during training.

This is the official PyTorch implementation of the following paper: [Traffic Forecasting on New Roads Using Spatial Contrastive Pre-Training](https://link.springer.com/article/10.1007/s10618-023-00982-0). In [**ECML PKDD SI DAMI 2023**](https://2023.ecmlpkdd.org/program/paper-session-overview/program-20-september-2023/). [[ArXiv](https://arxiv.org/abs/2305.05237)] [[Poster](/ECML_PKDD_traffic_poster_v3.pdf)] [[Slides](/ECMLPKDD23_unseen_roads_slides_v3_noGIF.pdf)] [[Talk](https://youtu.be/5urQyjyTyyM)] [[FigShare](https://figshare.com/s/ba3159f0a238a8c7a664)].

If you are interested in traffic forecasting, check out my collection of traffic forecasting papers: https://github.com/aprbw/traffic_prediction . Check also our previous paper, also focusing on node embeddings, but for spatial attention https://github.com/aprbw/G-SWaN/ .

# Abstract

New roads are being constructed all the time. However, the capabilities of previous deep forecasting models to generalize to new roads not seen in the training data (unseen roads) are rarely explored. In this paper, we introduce a novel setup called a spatio-temporal (ST) split to evaluate the models' capabilities to generalize to unseen roads. In this setup, the models are trained on data from a sample of roads, but tested on roads not seen in the training data. Moreover, we also present a novel framework called Spatial Contrastive Pre-Training (SCPT) where we introduce a spatial encoder module to extract latent features from unseen roads during inference time. This spatial encoder is pre-trained using contrastive learning. During inference, the spatial encoder only requires two days of traffic data on the new roads and does not require any re-training. We also show that the output from the spatial encoder can be used effectively to infer latent node embeddings on unseen roads during inference time. The SCPT framework also incorporates a new layer, named the spatially gated addition (SGA) layer, to effectively combine the latent features from the output of the spatial encoder to existing backbones. Additionally, since there is limited data on the unseen roads, we argue that it is better to decouple traffic signals to trivial-to-capture periodic signals and difficult-to-capture Markovian signals, and for the spatial encoder to only learn the Markovian signals. Finally, we empirically evaluated SCPT using the ST split setup on four real-world datasets. The results showed that adding SCPT to a backbone consistently improves forecasting performance on unseen roads. More importantly, the improvements are greater when forecasting further into the future.

# Requirements

```
intel-mkl==2020.3.304
python3==3.9.2
cuda==11.2.2
cudnn==8.1.1-cuda11
openmpi==4.1.0
magma==2.6.0
fftw3==3.3.8
pytorch==1.9.0

scipy==1.6.2
numpy==1.20.0
pandas==1.2.4
```

# Data

The datasets **METR-LA**, **PeMS-BAY**, and **PeMS-D7(m)** were obtained from the github repo of a recent benchmark study: https://github.com/deepkashiwa20/DL-Traff-Graph .

The PeMS-11k (our own naming) is from the GP-DCRNN github repo: https://github.com/tanwimallick/graph_partition_based_DCRNN . Since this dataset is 1 year long, and we are only interested in the spatial generalization, we make a shorter version of this dataset and call it **PeMS-11k(s)**. This is only 3 months long from Feburary 2018 to April 2018, inclusive.

# Run

Here are the explanations of the `.py` files:

`pred_GWN_16_adpAdj.py` the main file that runs the experiment.

`unseen_nodes.py` the spatiotemporal split.

`GWN_SCPT_14_adpAdj.py` the model.

`Metrics.py` metrics.

`Utils.py` some utility functions.

```
python3 ./pred_GWN_16_adpAdj.py \
1 `# 1: IS_PRETRN` \
.7 `# 2: R_TRN` \
0 `# 3: IS_EPOCH_1` \
42 `# 4: seed` \
100 `# 5: TEMPERATURE`\
METRLA `# 6: dataset` \
42 `# 7: seed_ss  spatial split` \
1 `# 8: IS_DESEASONED` \
0.0001 `# 9: weight_decay` \
1 `# 10: adp_adj` \
1 `# 11: is_SGA`
```

# Cite


```
Prabowo, A., Xue, H., Shao, W. et al. Traffic forecasting on new roads using spatial contrastive pre-training (SCPT). Data Min Knowl Disc (2023). https://doi.org/10.1007/s10618-023-00982-0
```


BibTex:

```
@article{prabowo2023SCPT,
  title={Traffic Forecasting on New Roads Unseen in the Training Data Using Spatial Contrastive Pre-Training},
  author={Prabowo, Arian and Xue, Hao and Shao, Wei and Koniusz, Piotr, and Salim, Flora D.},
  journal={Data Mining and Knowledge Discovery},
  year={2023},
  publisher={Springer}
}
```
