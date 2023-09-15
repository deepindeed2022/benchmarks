#!/bin/bash
#
#  Export model with max_batch_size = 16, opt batch_size = 16
#
#

TRT_VERSION=v$(python3 -c "import tensorrt as trt; print(trt.__version__)")
EXPORT_TRT_FP32=0
EXPORT_ONNX=1
bs=16
model_root="benchmark_bz${bs}_${TRT_VERSION}"
function test_export_cv() {
    local model_name=$1
    local img_size=$2
    model_path=models/yolov5s.pt
    if [[ -n $3 ]]; then
        model_path=$3
    fi
    echo $bs
    if [[ -n $4 ]]; then
        bs=$4
    fi
    model_dir=${model_root}/${model_name}_${img_size}
    if [[ ${model_name} == "yolov5s" ]]; do
        python3 -u export_onnx_models.py --model_name ${model_name} --model_path ${model_path} -bs ${bs} --img_size ${img_size} --simplify 
    else
        python3 -u export_onnx_models.py --model_name ${model_name} --model_path ${model_path} -bs ${bs} --img_size ${img_size} --simplify --do_constant_folding 
    done
    python3 -u build_engine.py --model_name ${model_name} --model_path ${model_name}_${img_size}_dynamic_bz${bs}_opset14.sim.onnx -bz ${bs} --image_size ${img_size} --fp16
    
    torch_path=${model_name}_${bs}_${img_size}.pth
    onnx_path=${model_name}_${img_size}_dynamic_bz${bs}_opset14.onnx
    trt_path=${model_name}_${img_size}_bz${bs}_trt_fp16_${TRT_VERSION}.engine

    mkdir -p $model_dir
    if [[ ${model_name} == "yolov5s" ]]; do
        cp configs/config_yolo.yml $model_dir/config.yml
    else
        cp configs/config.yml $model_dir/config.yml
    done
    sed -i "s/img_size/${img_size}/g"   ${model_dir}/config.yml
    sed -i "s/bs/${bs}/g"               ${model_dir}/config.yml
    sed -i "s/pthmodel/${torch_path}/g" ${model_dir}/config.yml
    sed -i "s/onnxmodel/${onnx_path}/g" ${model_dir}/config.yml
    sed -i "s/trtmodel/${trt_path}/g"   ${model_dir}/config.yml
    cp $model_dir/config.yml            $model_dir/config_torch.yml
    sed -i "s/\"TRTPredictor\"/\"TorchPredictor\"/g" ${model_dir}/config_torch.yml
    if [[ ${EXPORT_TRT_FP32} ]]; do
        python3 -u build_engine.py --model_name ${model_name} --model_path ${model_name}_${img_size}_dynamic_bz${bs}_opset14.sim.onnx -bz ${bs} --image_size ${img_size}
        cp $model_dir/config.yml $model_dir/config_fp32.yml
        sed -i "s/trt\_fp16/trt\_fp32/g" ${model_dir}/config_fp32.yml
    fi
    if [[ ${EXPORT_ONNX} ]]; do
        cp $model_dir/config.yml $model_dir/config_onnx.yml
        sed -i "s/\"TRTPredictor\"/\"OnnxPredictor\"/g"  ${model_dir}/config_onnx.yml
    fi
    mv ${model_name}_*.*  $model_dir/ 
}

function test_export_bert() {
    local model_name=$1
    local seq_length=$2
    bs=16
    if [[ -n $3 ]]; then
         bs=$3
    fi
    model_dir=${model_root}/${model_name}_${seq_length}
    rm -rf ${model_dir}

    python3 -u build_bert_engine.py --model_name ${model_name} -bz ${bs} --seq_length ${seq_length} --fp16

    torch_path=${model_name}_${seq_length}_${bs}.pth
    onnx_path=${model_name}_${seq_length}_dynamic_bz${bs}_opset14.onnx
    trt_path=${model_name}_${seq_length}_bz${bs}_trt_fp16_${TRT_VERSION}.engine

    mkdir -p $model_dir
    cp configs/config_bert.yml $model_dir/config.yml
    sed -i "s/seqlength/${seq_length}/g" ${model_dir}/config.yml
    sed -i "s/bs/${bs}/g" ${model_dir}/config.yml
    sed -i "s/pthmodel/${torch_path}/g" ${model_dir}/config.yml
    sed -i "s/onnxmodel/${onnx_path}/g" ${model_dir}/config.yml
    sed -i "s/trtmodel/${trt_path}/g" ${model_dir}/config.yml
    cp $model_dir/config.yml $model_dir/config_torch.yml
    sed -i "s/\"TRTPredictor\"/\"TorchPredictor\"/g" ${model_dir}/config_torch.yml
    if [[ ${EXPORT_TRT_FP32} ]]; do
        python3 -u build_bert_engine.py --model_name ${model_name} -bz ${bs} --seq_length ${seq_length}
        cp $model_dir/config.yml $model_dir/config_fp32.yml
        sed -i "s/trt\_fp16/trt\_fp32/g" ${model_dir}/config_fp32.yml
    fi

    if [[ ${EXPORT_ONNX} ]]; do
        cp $model_dir/config.yml $model_dir/config_onnx.yml
        sed -i "s/\"TRTPredictor\"/\"OnnxPredictor\"/g"  ${model_dir}/config_onnx.yml
    fi
    mv ${model_name}_*.*  $model_dir/ 
}

function download_models() {
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth -O models/yolox_s.pth
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth -O models/yolox_m.pth
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth -O models/yolox_l.pth
    wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/yolov5s.pt
    cd .. && git submodule update --init --recursive && cd -
}

function clean_model() {
    if [[ -d "output" ]]; then
        mkdir -p output
    fi
    mv *.onnx output/
    mv *.engine output/
}

test_export_cv resnet50 224

#for model in "resnet50" "resnext50_32x4d" "resnet101" "resnext101_32x8d" "resnext101_64x4d"; do
#   test_export_cv $model 224 
#done

# for model in "efficientnet-b0" "efficientnet-b3" "efficientnet-b5"; do
#    test_export_cv $model 224
# done

download_models
for bz in 16; do
 test_export_cv "yolov5s" 640  models/yolov5s.pt $bz
done

#test_export_cv "swinv2_tiny_window8_256" 256
#test_export_cv "swinv2_small_window8_256" 256
#test_export_cv "swinv2_base_window8_256" 256
#test_export_cv "swin_tiny_patch4_window7_224" 224
#test_export_cv "swin_small_patch4_window7_224" 224
#test_export_cv "swin_base_patch4_window7_224" 224
#test_export_cv "convnext_tiny" 224
#test_export_cv "convnext_small" 224
#test_export_cv "convnext_base" 224
#test_export_cv "vit_b_16" 224
#test_export_cv "vit_b_32" 224

## bert [1, 64] [16, 128] [32, 256]
#for model in "bert-base-uncased" "distilbert-base-uncased" "albert-base-v2"; do
#    for seqlen in 256; do
#	    test_export_bert $model $seqlen 16
#    done
#done