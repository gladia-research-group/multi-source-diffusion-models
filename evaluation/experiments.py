import functools
import json
from pathlib import Path
from typing import *

import torch
import yaml
from audio_diffusion_pytorch import KarrasSchedule

from main.data import ChunkedSupervisedDataset
from main.module_base import Model
from main.separation import *


ROOT_PATH = Path(__file__).parent.parent.resolve().absolute()


def stringify(obj:Union[Mapping,List,Tuple, Any]):
    if isinstance(obj, Mapping):
        return {k:stringify(v) for k,v in obj.items()}
    elif isinstance(obj, (List,Tuple)):
        return [stringify(v) for v in obj]
    else:
        return str(obj)
        
    
@torch.no_grad()
def separate_slakh_weak_msdm(
    model_paths: Mapping[str, str],
    dataset_path: str,
    output_dir: str,
    num_resamples: int = 1,
    num_steps: int = 150,
    batch_size: int = 16,
    resume: bool = True,
    device: float = torch.device("cuda:0"),
    s_churn: float = 20.0,
    use_gaussian: bool = False,
    source_id: Optional[int] = None,
    gamma: Optional[float] = None,
):
    config = stringify(locals())
    output_dir = Path(output_dir)

    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=22050,
        max_chunk_size=262144,
        min_chunk_size=262144,
    )

    if use_gaussian:
        assert gamma is not None
        diff_fn = functools.partial(differential_with_gaussian, gamma_fn=lambda s: gamma * s)
    else:
        assert source_id is not None
        diff_fn = functools.partial(differential_with_dirac, source_id=source_id)
    
    stem_to_model = {stem: Model.load_from_checkpoint(model_path).to(device) for stem, model_path in model_paths.items()}
    separator = WeaklyMSDMSeparator(
        stem_to_model=stem_to_model,
        sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=1.0, rho=7.0),
        differential_fn=diff_fn,
        s_churn=s_churn,
        num_resamples=num_resamples,
        use_tqdm=True,
    )
        
    separate_slakh(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
    )
    
    with open(output_dir/"config.yaml", "w") as f:
        yaml.dump(config, f)


@torch.no_grad()
def separate_slakh_msdm(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    num_resamples: int = 1,
    num_steps: int = 150,
    batch_size: int = 16,
    resume: bool = True,
    device: float = torch.device("cuda:0"),
    s_churn: float = 20.0,
    sigma_min: float = 1e-4,
    sigma_max: float = 1.0,
    use_gaussian: bool = False,
    source_id: Optional[int] = None,
    gamma: Optional[float] = None,
):
    config = stringify(locals())
    output_dir = Path(output_dir)

    dataset = ChunkedSupervisedDataset(
        audio_dir=dataset_path,
        stems=["bass", "drums", "guitar", "piano"],
        sample_rate=22050,
        max_chunk_size=262144,
        min_chunk_size=262144,
    )

    model = Model.load_from_checkpoint(model_path).to(device)

    if use_gaussian:
        assert gamma is not None
        diff_fn = functools.partial(differential_with_gaussian, gamma_fn=lambda s: gamma * s)
    else:
        assert source_id is not None
        diff_fn = functools.partial(differential_with_dirac, source_id=source_id)
    
    separator = MSDMSeparator(
        model=model,
        stems=["bass", "drums", "guitar", "piano"],
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        differential_fn=diff_fn,
        s_churn=s_churn,
        num_resamples=num_resamples,
        use_tqdm=True,
    )
        
    separate_slakh(
        output_dir=output_dir,
        dataset=dataset,
        separator=separator,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume,
    )
    
    with open(output_dir/"config.yaml", "w") as f:
        yaml.dump(config, f)


@torch.no_grad()
def separate_slakh(
        output_dir: Union[str, Path],
        dataset: SeparationDataset,
        separator: Separator,
        num_steps: int = 150,
        batch_size: int = 16,
        resume: bool = False,
    ):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create chunks metadata
    chunk_data = []
    for i in range(len(dataset)):
        start_sample, end_sample = dataset.get_chunk_indices(i)
        chunk_data.append(
            {
                "chunk_index": i,
                "track": dataset.get_chunk_track(i),
                "start_chunk_sample": start_sample,
                "end_chunk_sample": end_sample,
                "track_sample_rate": dataset.sample_rate,
                "start_chunk_seconds": start_sample / dataset.sample_rate,
                "end_chunk_in_seconds": end_sample / dataset.sample_rate,
            }
        )

    # Save chunk metadata
    with open(output_dir / "chunk_data.json", "w") as f:
        json.dump(chunk_data, f, indent=1)

    # Separate chunks
    separate_dataset(
        dataset=dataset,
        separator=separator,
        save_path=output_dir,
        num_steps=num_steps,
        batch_size=batch_size,
        resume=resume
    )
