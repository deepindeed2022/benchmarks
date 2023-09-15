import cv2
from typing import Tuple
import numpy as np
from transformers import AutoTokenizer


def image_preprocess(images, image_size=(224, 224)):
    outs = []
    for img in images:
        img = cv2.resize(
            img, image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img /= 255.0
        img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        outs.append(img)
    return np.array(outs)


class Feeder(object):
    def __init__(self, input_keys=["input"]):
        self._input_keys = input_keys

    def __call__(self, shape: Tuple = (1, 2), ignore_preprocess=True) -> dict:
        # assert len(shape) >= 1, "shape should >= 0"
        for i, e in enumerate(shape):
            assert e > 0, f"shape[{i}] should > 0, current shape[{i}]={e}"
        ret_dict = {}
        if ignore_preprocess:
            for key in self._input_keys:
                ret_dict[key] = np.random.rand(*shape).astype(np.float32)
            return ret_dict
        else:
            batch_size = shape[0]
            imgs = []
            for _ in range(batch_size):
                # image is (h,w,c) and input shape is (n,c,h,w),
                # when generate the image, we should convert input shape to image shape format
                imgs.append(np.random.randint(0, 255, size=(
                    shape[-2], shape[-1], shape[-3]), dtype=np.uint8))
            images = image_preprocess(imgs, image_size=shape[-2:])
            for key in self._input_keys:
                ret_dict[key] = images
            return ret_dict


class BertFeeder(Feeder):
    Text = "Tampilan : menarik Tampilan:menarik Warna : hitam Warna:hitam Bagus sekali tapi sayang kotaknya penyok padahal mau buat kado ulang tahun doi , jadi ngak estetik lagi deh , ntah dari sananya atau dari jasa pengiriman nya , kecewa banget sih , dengan harga segitu tapi ngak ngejamin sampai alamat dengan keadan yang baik . Bagus sekali tapi sayang kotaknya penyok padahal mau buat kado ulang tahun doi, jadi nggak estetik lagi deh, ntah dari sananya atau dari jasa pengiriman nya, kecewa banget sih, dengan harga segitu tapi nggak ngejamin sampai alamat dengan keadaan yang baik."

    def __init__(self, tokenizer_name="bert-base-uncased", input_keys=["input_ids", "attention_mask"], backend_type="TensorRTPredictor"):
        self._dtype = np.int64 if backend_type == "OnnxPredictor" else np.int32
        self._input_keys = input_keys
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, shape: Tuple = (1, 2), ignore_preprocess=True) -> dict:
        assert len(shape) == 2
        batch_size, seq_length = shape[0], shape[1]
        # assert seq_length > 0, "seqlength should > 0"
        # assert batch_size >= 1, "batch_size should >= 1"
        text = BertFeeder.Text
        tinputs = self._tokenizer(text.lower(
        ), truncation=True, padding="max_length", max_length=seq_length, return_tensors="pt")
        inputs = {key: tinputs[key].numpy() for key in tinputs}
        # TODO:
        # current only match on text sequence bert model, if the model input with multi token_type, the
        # "token_type_ids" result is mismatched
        if "token_type_ids" in self._input_keys:
            inputs["token_type_ids"] = np.ones(shape=[1, seq_length])
        #
        # only support for absolute distance
        if "position_ids" in self._input_keys:
            inputs["position_ids"] = np.arange(
                0, seq_length, dtype=np.int32).reshape(1, seq_length)
        # Because there is a Tile OP for the position embedding in the model,
        # the batch size of position_ids in the input should be 1
        if batch_size > 1:
            for inp in inputs:
                inputs[inp] = np.repeat(inputs[inp], batch_size, axis=0)
        for k, v in inputs.items():
            inputs[k] = v.astype(self._dtype)
        return inputs


def test_feeder():
    feed = Feeder()
    inputs = feed(shape=(1, 2))
    assert "input" in inputs
    inputs = feed(shape=(2, 3, 5, 5))
    print(inputs["input"].shape)
    inputs = feed(shape=(2, 3, 5, 5), ignore_preprocess=False)
    print(inputs["input"].shape)


def test_textfeeder():
    keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids"]
    bertfeed = BertFeeder(input_keys=keys)
    inputs = bertfeed(shape=(2, 256))

    for k, v in inputs.items():
        assert k in keys
        print(v.shape)


if __name__ == "__main__":
    test_feeder()
    test_textfeeder()
