# CUDA_VISIBLE_DEVICES=1 python baseline.py "data/aircraft,data/cub200" \
# -d "Aircraft,CUB200" -sr 100 --seed 0 \
# --epochs 500 --lr-patience 10 --workers 10 --batch-size 32 --log "logs/" \
# --arch "AutoGrow" \
# --pretrained "/DATA/arjun_ashok/files/lifelong/avalanche-veniat/autogrow/results/2021-09-01_06-52-41/_best_ckpt.t7" \
# --run-name "CIFAR100-Plain-K50"

# CUDA_VISIBLE_DEVICES=1 python baseline.py "data/cub200,data/stanford_cars,data/aircraft" \
# -d "CUB200,StanfordCars,Aircraft" -sr 100 --seed 0 \
# --epochs 500 --lr-patience 10 --workers 10 --batch-size 16 --log "logs/" \
# --arch "AutoGrow" \
# --pretrained "/DATA/arjun_ashok/files/lifelong/avalanche-veniat/autogrow/results/2021-09-01_04-26-55/_best_ckpt.t7" \
# --run-name "CIFAR100-Plain-K10"

CUDA_VISIBLE_DEVICES=1 python baseline.py "data/cub200,data/stanford_cars,data/aircraft" \
-d "CUB200,StanfordCars,Aircraft" -sr 100 --seed 0 \
--epochs 500 --lr-patience 10 --workers 10 --batch-size 16 --log "logs/" \
--arch "AutoGrow" \
--pretrained "/DATA/arjun_ashok/files/lifelong/avalanche-veniat/autogrow/results/CIFAR100-Plain-K10-BASELINE-LRF-0.1-2021-09-02_15-57-07/baseline/_best_ckpt.t7" \
--run-name "CIFAR100-Plain-K10-BASELINE"