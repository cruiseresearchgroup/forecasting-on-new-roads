# Traffic Forecasting on New Roads Unseen in the Training Data Using Spatial Contrastive Pre-Training

![Visual_abstract](/vizabs2.png) Our novel traffic forecasting framework, Spatial Contrastive Pre-Training (SCPT), enables accurate forecasts on new roads (orange) that were not seen during training.

This is the official PyTorch implementation of the following paper: Traffic Forecasting on New Roads Unseen in the Training Data Using Spatial Contrastive Pre-Training. In ECML PKDD SI DAMI. [[ArXiv](https://arxiv.org/abs/2302.09956)].

If you are interested in traffic forecasting, check out my collection of traffic forecasting papers: https://github.com/aprbw/traffic_prediction.

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

Harvard:

BibTex:
