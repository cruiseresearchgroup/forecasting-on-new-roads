# forecasting-on-new-roads-2023-05

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

# Run

```
python3 ./pred_GWN_16_adpAdj.py \
1 `# 1: IS_PRETRN` \
.1 `# 2: R_TRN` \
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


