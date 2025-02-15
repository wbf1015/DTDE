bash myrun.sh 0 2200 512 4.0 4.0 0.02 150 32 32 1007 models/RotatE-RotatE-FB15k237-256 FB15k-237 1 1 Adam MultiStepLR 32 \
    -pretrain_path2 models/LorentzKG-RotatE-FB15k-237-256 -warm_up_epochs 100 \
    -temprature 0.5 -kdloss_weight 1.0 -temprature_ts 0.5 -pre_sample_size 50 \
    -contrastive_tau 1.5 -contrastive_weight 0.01 \
    -add_bias

bash myrun.sh 0 2200 512 9.0 9.0 0.02 150 32 32 1008 models/RotatE-RotatE-FB15k237-256 FB15k-237 1 1 Adam MultiStepLR 32 \
    -pretrain_path2 models/LorentzKG-RotatE-FB15k-237-256 -warm_up_epochs 100 \
    -temprature 0.5 -kdloss_weight 1.0 -temprature_ts 0.5 -pre_sample_size 50 \
    -contrastive_tau 1.5 -contrastive_weight 0.01 \
    -add_bias

bash myrun.sh 0 2200 512 12.0 12.0 0.02 150 32 32 1009 models/RotatE-RotatE-FB15k237-256 FB15k-237 1 1 Adam MultiStepLR 32 \
    -pretrain_path2 models/LorentzKG-RotatE-FB15k-237-256 -warm_up_epochs 100 \
    -temprature 0.5 -kdloss_weight 1.0 -temprature_ts 0.5 -pre_sample_size 50 \
    -contrastive_tau 1.5 -contrastive_weight 0.01 \
    -add_bias

