from collections import defaultdict
import json
import os
from pathlib import Path
from pathlib import Path
from typing import *
import math

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from torchaudio.functional import resample

from main.data import is_silent


def sdr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    s_target = torch.norm(target, dim=-1)**2 + eps
    s_error = torch.norm(target - preds, dim=-1)**2 + eps
    return 10 * torch.log10(s_target/s_error)


def sisnr(preds: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target
    noise = target_scaled - preds
    s_target = torch.sum(target_scaled**2, dim=-1) + eps
    s_error = torch.sum(noise**2, dim=-1) + eps
    return 10 * torch.log10(s_target / s_error)


def load_chunks(chunk_folder: Path, stems: Sequence[str]) -> Tuple[Mapping[str, torch.Tensor], int]:
    separated_tracks_and_rate = {s: torchaudio.load(chunk_folder / f"{s}.wav") for s in stems}
    separated_tracks = {k:t for k, (t,_) in separated_tracks_and_rate.items()}
    sample_rates_sep = [s for (_,s) in separated_tracks_and_rate.values()]

    assert len({*sample_rates_sep}) == 1, print(sample_rates_sep)
    sr = sample_rates_sep[0]

    return separated_tracks, sr


def load_and_resample_track(track_path: Union[str,Path], stems: Sequence[str], resample_sr: int) -> Mapping[str, torch.Tensor]:
    track_path = Path(track_path)
    
    def _load_and_resample_stem(stem: Path):
        stem_path = track_path/f"{stem}.wav"
        if stem_path.exists():
            wav, sr = torchaudio.load(stem_path)
            return resample(wav, sr, resample_sr)
        else:
            return None
    
    # Load and resample track stems
    stem_to_track = {s:_load_and_resample_stem(s) for s in stems}
    
    # Assert it contains at least a single source
    assert set(stem_to_track.values()) != {None}
    
    # Get sources dimensionality
    shapes = {wav.shape for s, wav in stem_to_track.items() if wav is not None}
    num_channels = {channels for (channels,length) in shapes}
    sample_lengths = {length for (channels,length) in shapes} 
    
    # Assert the existing sources have same dimensionality (up to certaian threshold)
    assert len(num_channels) == 1, f"{num_channels}"
    num_channels, = num_channels
    assert max(sample_lengths) - min(sample_lengths) <= 0.1 * resample_sr, f"{(max(sample_lengths) - min(sample_lengths))/resample_sr}"
    
    for s, wav in stem_to_track.items():
        # Initialize missing sources to zero
        if wav is None:
            stem_to_track[s] = torch.zeros(size=(num_channels, min(sample_lengths)) )
        
        # Crop sources
        else:
            stem_to_track[s] = stem_to_track[s][:,:min(sample_lengths)]
    
    return stem_to_track
    

def get_full_tracks(
    separation_path: Union[str, Path],
    expected_sample_rate: int = 22050,
    stems: Sequence[str] = ("bass","drums","guitar","piano"),
):
    separation_folder = Path(separation_path)
    assert separation_folder.exists(), separation_folder
    assert (separation_folder / "chunk_data.json").exists(), separation_folder

    with open(separation_folder / "chunk_data.json") as f:
        chunk_data = json.load(f)

    track_to_chunks = defaultdict(list)
    for chunk_data in chunk_data:
        track = chunk_data["track"]
        chunk_idx = chunk_data["chunk_index"]
        start_sample = chunk_data["start_chunk_sample"]
        #track_sample_rate = chunk_data["sample_rate"]
        track_to_chunks[track].append( (start_sample, chunk_idx) )

    # Reorder chunks into ascending order
    for track, chunks in tqdm(track_to_chunks.items()):
        sorted_chunks = sorted(chunks)
        
        # Merge separations
        separated_wavs = {s: [] for s in stems}
        for _, chunk_idx in sorted_chunks:
            chunk_folder = separation_folder / str(chunk_idx)
            
            separated_chunks, sr = load_chunks(chunk_folder, stems)
            assert sr == expected_sample_rate, f"{sr} different from expected sample-rate {expected_sample_rate}"
            
            # convert start sample to the chunk sample-rate
            #start_sample = start_sample * sr // track_sample_rate

            for s in separated_chunks:
                separated_wavs[s].append(separated_chunks[s])

        for s in stems:
            separated_wavs[s] = torch.cat(separated_wavs[s], dim=-1)
            
        yield track, separated_wavs


def save_tracks(
    separation_path: Union[str, Path],
    output_path: Union[str,Path],
    expected_sample_rate: int = 22050,
    stems=("bass","drums","guitar","piano"),
):
    os.mkdir(output_path)
    for track, stem_to_wav in get_full_tracks(separation_path, expected_sample_rate, stems):
        os.mkdir(output_path / track)
        for s,w in stem_to_wav.items():
            torchaudio.save(output_path / track/f"{s}.wav", w.cpu(), sample_rate=expected_sample_rate)


def evaluate_separations(
    separation_path: Union[str, Path],
    dataset_path: Union[str, Path],
    separation_sr: int,
    filter_single_source: bool = True,
    stems: Sequence[str] = ("bass","drums","guitar","piano"),
    eps: float = 1e-8,
    chunk_duration: float = 4.0, 
    overlap_duration: float = 2.0
) -> pd.DataFrame:

    separation_path = Path(separation_path)
    dataset_path = Path(dataset_path)
    
    df_entries = defaultdict(list)
    for track, separated_track in get_full_tracks(separation_path, separation_sr, stems):
        
        # load and resample track
        original_track = load_and_resample_track(dataset_path/track, stems, 22050)

        # Adjust for changes in length
        for s in stems:
            max_length = separated_track[s].shape[-1]
            original_track[s] = original_track[s][:,:max_length]
        
        # Compute mixture
        mixture = sum([owav for owav in original_track.values()])

        chunk_samples = int(chunk_duration * separation_sr)
        overlap_samples = int(overlap_duration * separation_sr)

        # Calculate the step size between consecutive sub-chunks
        step_size = chunk_samples - overlap_samples

        # Determine the number of evaluation chunks based on step_size
        num_eval_chunks = math.ceil((mixture.shape[-1] - overlap_samples) / step_size)
            
        for i in range(num_eval_chunks):
            start_sample = i * step_size
            end_sample = start_sample + chunk_samples
            
            # Determine number of active signals in sub-chunk
            num_active_signals = 0
            for k in separated_track:
                o = original_track[k][:,start_sample:end_sample]
                if not is_silent(o):
                    num_active_signals += 1
            
            # Skip sub-chunk if necessary
            if filter_single_source and num_active_signals <= 1:
                continue

            # Compute SI-SNRi for each stem
            for k in separated_track:
                o = original_track[k][:,start_sample:end_sample]
                s = separated_track[k][:,start_sample:end_sample]
                m = mixture[:,start_sample:end_sample]
                df_entries[k].append((sisnr(s, o, eps) - sisnr(m, o, eps)).item())
            
            # Add chunk and sub-chunk info to dataframe entry
            df_entries["start_sample"].append(start_sample)
            df_entries["end_sample"].append(end_sample)

    # Create and return dataframe
    return pd.DataFrame(df_entries)
