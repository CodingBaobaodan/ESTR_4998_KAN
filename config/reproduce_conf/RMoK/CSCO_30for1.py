exp_conf = dict(
    model_name="DenseRMoK",
    dataset_name='CSCO',     # Set to CSCO to point to your new dataset

    hist_len=60,
    pred_len=1,

    revin_affine=False,       # Retain other configurations as in ETTh1

    lr=0.001,                 # Learning rate
)
