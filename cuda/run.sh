# ncu --nvtx --nvtx-include "[Infer" --force-overwrite -o resnet50 \
# python3 -u eval_engine.py ../model_converts/benchmark_bz16_v8.6.1/resnet50_224/resnet50_224_bz16_trt_fp16_v8.6.1.engine
# ncu -o resnet50 --replay-mode range --force-overwrite --set roofline \
# python3 -u eval_engine.py ../model_converts/benchmark_bz16_v8.6.1/resnet50_224/resnet50_224_bz16_trt_fp16_v8.6.1.engine

ncu -o resnet50_fp32 --replay-mode range --force-overwrite --set roofline \
python3 -u eval_engine.py ../model_converts/benchmark_bz16_v8.6.1/resnet50_224/resnet50_224_bz16_trt_fp32_v8.6.1.engine
