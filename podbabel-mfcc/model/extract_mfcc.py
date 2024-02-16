import torch
import torchaudio
import json
import math
import numpy as np
from torch.utils.mobile_optimizer import optimize_for_mobile

default_sample_rate = 44100
n_mfcc = 2
window_length = int(math.pow(2, 16))
n_fft = window_length
hop_length = int(window_length / 4)
n_mels = 23

melkwargs = {
    "n_fft": n_fft,
    "win_length": window_length,
    "hop_length": hop_length,
    "n_mels": n_mels,
    "center": False,
}


class MFCCModel(torch.nn.Module):

    def __init__(self, default_sample_rate = default_sample_rate, n_mfcc: int = n_mfcc, melkwargs = melkwargs):
        super(MFCCModel, self).__init__()

        self.default_sample_rate = default_sample_rate


        sample_rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
        sample_rates = [rate for rate in sample_rates if rate != default_sample_rate]

        for rate in sample_rates:
            setattr(self, f'resampler{rate}', torchaudio.transforms.Resample(rate, default_sample_rate))

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=default_sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=melkwargs
        )

    def forward(self, waveform: torch.Tensor, config: torch.Tensor) -> torch.Tensor:
        orig_sample_rate = config[0].item()

            # 转换为单声道
        waveform = waveform.mean(dim=0)

        if orig_sample_rate == 8000:
            waveform = self.resampler8000(waveform)
        elif orig_sample_rate == 11025:
            waveform = self.resampler11025(waveform)
        elif orig_sample_rate == 12000:
            waveform = self.resampler12000(waveform)
        elif orig_sample_rate == 16000:
            waveform = self.resampler16000(waveform)
        elif orig_sample_rate == 22050:
            waveform = self.resampler22050(waveform)
        elif orig_sample_rate == 24000:
            waveform = self.resampler24000(waveform)
        elif orig_sample_rate == 32000:
            waveform = self.resampler32000(waveform)
        # elif orig_sample_rate == 44100:
        #     waveform = self.resampler44100(waveform)
        elif orig_sample_rate == 48000:
            waveform = self.resampler48000(waveform)
        elif orig_sample_rate == 96000:
            waveform = self.resampler96000(waveform)

        return self.mfcc_transform(waveform)

# 创建模型
model = MFCCModel()

# 将模型转换为TorchScript
scripted_model = torch.jit.script(model)
optimized_model = optimize_for_mobile(scripted_model)
optimized_model_metal = optimize_for_mobile(scripted_model, backend='metal')
print(torch.jit.export_opnames(optimized_model))

# 保存TorchScript模型
scripted_model.save("mfcc_model.pt")
optimized_model._save_for_lite_interpreter('./mfcc_model.ptl')
optimized_model_metal._save_for_lite_interpreter('./mfcc_model_metal.ptl')

loaded_model = torch.jit.load("mfcc_model.pt")

def extract_mfcc(filename: str, model: torch.nn.Module, json_filename: str):
    waveform, orig_sample_rate = torchaudio.load(filename)

    mfcc: torch.Tensor = model(waveform, torch.tensor([orig_sample_rate]) )
    mfcc = mfcc.to('cpu').float()

    mfcc_numpy: np.ndarray  = mfcc.detach().numpy()

    print(mfcc_numpy.shape)

    mfcc_list = mfcc_numpy.T.tolist()

    with open(json_filename, 'w') as f:
        json.dump(mfcc_list, f)

#extract_mfcc("stereo44100 2.mp3", loaded_model, "mfcc_1.json")
