import abc
from pathlib import Path
from typing import List, Optional, Callable, Mapping

import torch
import torchaudio
import tqdm
from math import sqrt, ceil

from audio_diffusion_pytorch.diffusion import Schedule
from torch.utils.data import DataLoader

from main.data import assert_is_audio, SeparationDataset
from main.module_base import Model

class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate(mixture, num_steps) -> Mapping[str, torch.Tensor]:
        ...
    
    
class MSDMSeparator(Separator):
    def __init__(self, model: Model, stems: List[str], sigma_schedule: Schedule, **kwargs):
        super().__init__()
        self.model = model
        self.stems = stems
        self.sigma_schedule = sigma_schedule
        self.separation_kwargs = kwargs
    
    def separate(self, mixture: torch.Tensor, num_steps:int = 100):
        device = self.model.device
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape
        
        y = separate_mixture(
            mixture=mixture,
            denoise_fn=self.model.model.diffusion.denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(self.stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem:y[:,i:i+1,:] for i,stem in enumerate(self.stems)}


class WeaklyMSDMSeparator(Separator):
    def __init__(self, stem_to_model: Mapping[str, Model], sigma_schedule, **kwargs):
        super().__init__()
        self.stem_to_model = stem_to_model
        self.separation_kwargs = kwargs
        self.sigma_schedule = sigma_schedule

    def separate(self, mixture: torch.Tensor, num_steps: int):
        stems = self.stem_to_model.keys()
        models = [self.stem_to_model[s] for s in stems]
        fns = [m.model.diffusion.denoise_fn for m in models]
        
        # get device of models
        devices = {m.device for m in models}
        assert len(devices) == 1, devices
        (device,) = devices
        
        mixture = mixture.to(device)
        batch_size, _, length_samples = mixture.shape

        def denoise_fn(x, sigma):
            xs = [x[:, i:i+1] for i in range(4)]
            xs = [fn(x,sigma=sigma) for fn,x in zip(fns, xs)]
            return torch.cat(xs, dim=1)
        
        y = separate_mixture(
            mixture=mixture,
            denoise_fn=denoise_fn,
            sigmas=self.sigma_schedule(num_steps, device),
            noises=torch.randn(batch_size, len(stems), length_samples).to(device),
            **self.separation_kwargs,
        )
        return {stem:y[:,i:i+1,:] for i, stem in enumerate(stems)}


# Algorithms ------------------------------------------------------------------

def differential_with_dirac(x, sigma, denoise_fn, mixture, source_id=0):
    num_sources = x.shape[1]
    x[:, [source_id], :] = mixture - (x.sum(dim=1, keepdim=True) - x[:, [source_id], :])
    score = (x - denoise_fn(x, sigma=sigma)) / sigma
    scores = [score[:, si] for si in range(num_sources)]
    ds = [s - score[:, source_id] for s in scores]
    return torch.stack(ds, dim=1)


def differential_with_gaussian(x, sigma, denoise_fn, mixture, gamma_fn=None):
    gamma = sigma if gamma_fn is None else gamma_fn(sigma)
    d = (x - denoise_fn(x, sigma=sigma)) / sigma 
    d = d - sigma / (2 * gamma ** 2) * (mixture - x.sum(dim=[1], keepdim=True)) 
    #d = d - 8/sigma * (mixture - x.sum(dim=[1], keepdim=True)) 
    return d


@torch.no_grad()
def separate_mixture(
    mixture: torch.Tensor, 
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    noises: Optional[torch.Tensor],
    differential_fn: Callable = differential_with_dirac,
    s_churn: float = 0.0, # > 0 to add randomness
    num_resamples: int = 1,
    use_tqdm: bool = False,
):      
    # Set initial noise
    x = sigmas[0] * noises # [batch_size, num-sources, sample-length]
    
    vis_wrapper  = tqdm.tqdm if use_tqdm else lambda x:x 
    for i in vis_wrapper(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i+1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

            # Compute conditioned derivative
            d = differential_fn(mixture=mixture, x=x, sigma=sigma_hat, denoise_fn=denoise_fn)

            # Update integral
            x = x + d * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x)
    
    return x.cpu().detach()

# -----------------------------------------------------------------------------
def save_separation(
    separated_tracks: Mapping[str, torch.Tensor],
    sample_rate: int,
    chunk_path: Path,
):    
    for stem, separated_track in separated_tracks.items():
        assert_is_audio(separated_track)
        torchaudio.save(chunk_path / f"{stem}.wav", separated_track.cpu(), sample_rate=sample_rate)


@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    num_steps: int,
    save_path: str,
    resume: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
):
    
    save_path = Path(save_path)
    if not resume and save_path.exists() and not len(list(save_path.glob("*"))) == 0:
        raise ValueError(f"Path {save_path} already exists!")

    # Get samples
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Main separation loop
    chunk_id = 0
    for batch_idx, batch in enumerate(loader):
        last_chunk_batch_id = chunk_id + batch[0].shape[0] - 1
        chunk_path = save_path / f"{last_chunk_batch_id}"
        
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            chunk_id = chunk_id + batch[0].shape[0]
            continue

        print(f"{chunk_id=}")
        tracks = [b for b in batch]
        print(f"batch {batch_idx+1} out of {ceil(len(dataset) / batch[0].shape[0])}")
        
        # Generate mixture
        mixture = sum(tracks)
        seps_dict = separator.separate(mixture=mixture, num_steps=num_steps)

        # Save separated audio
        num_samples = tracks[0].shape[0]
        for i in range(num_samples):
            chunk_path = save_path / f"{chunk_id}"
            chunk_path.mkdir(parents=True, exist_ok=True)
            
            save_separation(
                separated_tracks={stem: sep[i] for stem, sep in seps_dict.items()},
                sample_rate=dataset.sample_rate,
                chunk_path=chunk_path,
            )
            chunk_id += 1


