import sys
import torch
import argparse
import timm
from efficientnet_pytorch import EfficientNet
from transformers import AutoTokenizer, AutoModel

from model_utils import setup_logger, test_infer_performance, test_bert_infer_performance


def main(model_name, num_data, batch_size, use_mixed_precision, max_seq_length=512):
    if model_name == 'resnet50':
        # slow download
        # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        test_infer_performance(model=model, model_name=model_name,
                               batch_size=batch_size, num_data=num_data, use_mixed_precision=use_mixed_precision)
    elif model_name == 'yolov5':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        test_infer_performance(model=model, model_name=model_name,
                               batch_size=batch_size, num_data=num_data, input_shape=(
                                   3, 640, 640),
                               use_mixed_precision=use_mixed_precision)
    elif model_name == "yolov7x":
        sys.path.append("./models")
        import yolox.models.yolox as yolox
        from model_utils import model_locations
        model = torch.load(model_locations[model_name])

        test_infer_performance(model=model, model_name=model_name,
                               batch_size=batch_size, num_data=num_data, input_shape=(
                                   3, 640, 640),
                               use_mixed_precision=use_mixed_precision)
    elif model_name == "yolox":
        sys.path.append("./models")
        from yolox.models.yolox import YOLOX
        model = YOLOX()
        test_infer_performance(model=model, model_name=model_name, batch_size=batch_size,
                               num_data=num_data, input_shape=(3, 640, 640), use_mixed_precision=use_mixed_precision)

    elif model_name == "yolox-l-3w" or model_name == "yolox-l-4w":
        sys.path.append("./models")
        import yolox.models.yolox as yolox
        from model_utils import model_locations
        model = torch.load(model_locations[model_name], map_location="cpu")

        test_infer_performance(model=model, model_name=model_name, batch_size=batch_size, input_shape=(3, 640, 640),
                               num_data=num_data, use_fp16=True, use_mixed_precision=use_mixed_precision)
    elif model_name == "efficientnet-b4":
        model = EfficientNet.from_pretrained(model_name)
        model.set_swish(memory_efficient=False)
        test_infer_performance(model=model, model_name=model_name, batch_size=batch_size,
                               num_data=num_data, use_mixed_precision=use_mixed_precision)
    elif model_name == 'swin' or model_name == 'swin_transformer':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True)
        test_infer_performance(model=model, model_name=model_name,
                               batch_size=batch_size, num_data=num_data, use_mixed_precision=use_mixed_precision)
    elif model_name == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        test_bert_infer_performance(model=model, tokenizer=tokenizer, model_name=model_name,
                                    batch_size=batch_size, num_data=num_data, max_length=max_seq_length,
                                    use_mixed_precision=use_mixed_precision)
    else:
        raise TypeError(f'Wrong model name: {model_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        required=True, help='model to run benchmark')
    parser.add_argument('-nd', '--num_data', type=int,
                        default=10240, help='num of data')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument("--seq_length", type=int, default=512,
                        help="bert model max seq length")
    parser.add_argument('-mp', '--mixed_precision', action='store_true')
    args = parser.parse_args()
    setup_logger(logname="run_onnx.log")
    main(args.model, args.num_data, args.batch_size, args.mixed_precision)
