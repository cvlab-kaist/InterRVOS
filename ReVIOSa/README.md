# Training & Inference

## Environment

We provide a pre-built Docker image on Docker Hub, available [here](https://hub.docker.com/r/woojeongjin/interrvos):

```bash
docker pull woojeongjin/interrvos:latest
```

- PyTorch: 2.3.1
- CUDA: 12.1

## Training

<details open>
<summary><b>Pretrained Model Preparation</b></summary>

You are expected to download the following pretrained models and place them in the `./pretrained_weights` directory:
- [sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large)
- [InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B)
- [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)

```
pretrained_weights/
├── sam2_hiera_large.pt
├── InternVL2_5-1B
└── InternVL2_5-4B
```
</details>

<br>

<details open>
<summary><b>Data Preparation</b></summary>

We are planning to release InterRVOS-127K, so be please stay tuned!

The final data structure should be:
```
datasets
├── InterRVOS
├── mevis
├── ref_ytbvos
└── ref_davis
```
To simplify the code, we unify the dataset format as in [MeViS](https://github.com/henghuiding/MeViS), each dataset must contain `meta_expressions.json` and `mask_dict.json` files consistent with the MeViS structure.
Ref-Youtube-VOS and Ref-DAVIS (modified versions) will be made available alongside InterRVOS-127K.
</details>

### Scripts

You can train with the following script:

```bash
# Adjust the config file and the number of GPUs
bash tools/dist.sh train model/configs/$CONFIG.py $N_GPU
```
Or, you can simply run by using:
```bash
bash scripts/train.sh reviosa_1b 4
```

## Inference

<details open>
<summary><b>Convert model to Huggingface format</b></summary>

Before running inference, you need to convert the trained model checkpoint:
```bash
export PYTHONPATH=$(pwd)

python model/hf/convert_to_hf.py model/configs/$CONFIG.py \
    --pth-model $TARGET_PATH \
    --save-path $SAVE_PATH
```
- `$TARGET_PATH`: path to the trained .pth checkpoint
- `$SAVE_PATH`: output directory where the huggingface model will be saved
</details>

### Scripts

You can run inference with the following command:
```bash
export CUDA_VISIBLE_DEVICES=0

# Adjust EXP_NAME and pid settings
HF_PATH="wooj0216/ReVIOSa-4B"
python model/evaluation/ref_vos_eval.py \
    $HF_PATH \
    --dataset INTERRVOS \
    --work_dir EVAL/$EXP_NAME/interrvos \
    --n_pid $N_PID --pid $PID
```
Or, you can simply run by using:
```bash
bash scripts/inference_interrvos.sh 2 0 3 reviosa_1b
```
This example will run inference with:
- Model: ReVIOSa/HF/reviosa_1b
- GPU: index 2
- Splits: `pid=0` among a total of 3 splits
- Results : `results_pid_0.json` (you need to additionally run `pid=1` and `pid=2` to obtain the complete inference results)

## Evaluation

After obtaining `results_pid_{pid}.json` files from [inference stage](#inference), please run the following script to merge and evaluate:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Merge split results
python tools/eval/merge_json.py \
    --result_dir EVAL/$EXP_NAME/interrvos \
    --save_path EVAL/$EXP_NAME/interrvos/results.json

# Run evaluation
python tools/eval/eval_interrvos.py \
    --pred_path EVAL/$EXP_NAME/interrvos/results.json \
    --data_dir datasets/InterRVOS/val \
    --save_path EVAL/$EXP_NAME/interrvos/eval_actor.json
```
You can also find inference and evaluation scripts for other datasets in the scripts directory.

Note that our model produces segmentation masks not only for the <i>actor objects</i> but also for the <i>target objects</i>. All relevant scripts are provided in `scripts`.
