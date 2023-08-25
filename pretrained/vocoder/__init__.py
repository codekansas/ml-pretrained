from typing import Literal, cast, get_args, overload

from .hifigan import HiFiGAN, PretrainedHiFiGANType, pretrained_hifigan
from .waveglow import WaveGlow, pretrained_waveglow

VocoderType = Literal["waveglow", "hifigan"]
Vocoder = HiFiGAN | WaveGlow


def cast_vocoder_type(vocoder_type: str) -> VocoderType:
    args = get_args(VocoderType)
    assert vocoder_type in args, f"Unknown vocoder type: {vocoder_type}"
    return cast(VocoderType, vocoder_type)


@overload
def pretrained_vocoder(vocoder_type: Literal["hifigan"], *, hifigan: PretrainedHiFiGANType = "16000hz") -> HiFiGAN:
    ...


@overload
def pretrained_vocoder(vocoder_type: Literal["waveglow"]) -> WaveGlow:
    ...


def pretrained_vocoder(vocoder_type: VocoderType, *, hifigan: PretrainedHiFiGANType = "16000hz") -> Vocoder:
    match vocoder_type:
        case "hifigan":
            return pretrained_hifigan(hifigan)
        case "waveglow":
            return pretrained_waveglow()
        case _:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
