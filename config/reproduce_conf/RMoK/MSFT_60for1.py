exp_conf = dict(
    model_name="DenseRMoK",
    dataset_name='MSFT',     # Set to MSFT to point to your new dataset

    hist_len=60,
    pred_len=1,

    revin_affine=True,       # Retain other configurations as in ETTh1

    lr=0.001,                 # Learning rate
)
