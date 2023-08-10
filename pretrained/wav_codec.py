"""Waveform codec model."""

import argparse
import logging

import torch
import torch.nn.functional as F
import torchaudio
from ml.models.codebook import ResidualVectorQuantization, VectorQuantization
from ml.utils.device.auto import detect_device
from ml.utils.logging import configure_logging
from torch import Tensor, nn, optim

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    __constants__ = ["stride_length"]

    def __init__(self, stride_length: int, hidden_size: int) -> None:
        super().__init__()

        self.stride_length = stride_length

        self.in_proj = nn.Sequential(
            nn.Linear(stride_length, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # self.rnn = nn.LSTM(stride_length, hidden_size, num_layers=2, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unflatten(-1, (-1, self.stride_length))
        # x, _ = self.rnn(x)
        x = self.in_proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, stride_length: int, hidden_size: int) -> None:
        super().__init__()

        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, stride_length)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x = self.out_proj(x)
        x = x.flatten(-2)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        stride_length: int = 320,
        hidden_size: int = 512,
        codebook_size: int = 1024,
        num_quantizers: int = 4,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(stride_length, hidden_size)
        self.decoder = Decoder(stride_length, hidden_size)
        self.rvq = ResidualVectorQuantization(
            VectorQuantization(dim=hidden_size, codebook_size=codebook_size),
            num_quantizers=num_quantizers,
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        x: Tensor = self.encoder(waveform)
        x, _, codebook_loss, _ = self.rvq(x.transpose(1, 2))
        x = self.decoder(x.transpose(1, 2))
        return x, codebook_loss


def test_codec_adhoc() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Runs adhoc test of the codec.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument("output_file", type=str, help="Path to output audio file")
    parser.add_argument("-n", "--num-steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("-l", "--log-interval", type=int, default=1, help="Log interval")
    parser.add_argument("-s", "--num-seconds", type=float, default=5.0, help="Number of seconds to use")
    args = parser.parse_args()

    # Loads the audio file.
    audio, sr = torchaudio.load(args.input_file)
    audio = audio[:1]
    audio = audio[:, : int(sr * args.num_seconds)]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Note: This normalizes the audio to the range [-1, 1], which may increase
    # the volume of the audio if it is quiet.
    audio = audio / audio.abs().max() * 0.999

    device = detect_device()
    audio = device.tensor_to(audio)

    # Loads the model.
    model = Autoencoder()
    model.to(device._get_device())

    # Runs training.
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    with device.autocast_context():
        for i in range(args.num_steps):
            opt.zero_grad()
            rec_audio, codebook_loss = model(audio)
            loss = F.l1_loss(rec_audio, audio) + codebook_loss.sum()
            if torch.isnan(loss).any():
                logger.warning("NaN loss; aborting")
                break
            loss.backward()
            opt.step()

            if i % args.log_interval == 0:
                logger.info("Step %d: loss=%f", i, loss.item())

        rec_audio, _ = model(audio)
        rec_audio = rec_audio.detach().cpu().float()

    # Saves the audio.
    torchaudio.save(args.output_file, rec_audio, 16000)

    logger.info("Saved %s", args.output_file)


if __name__ == "__main__":
    # python -m pretrained.wav_codec
    test_codec_adhoc()
