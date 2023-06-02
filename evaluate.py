from pathlib import Path
import hydra
from omegaconf import DictConfig

from evaluation.evaluate_separation import evaluate_separations
from main import utils


log = utils.get_logger(__name__)


@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig):
    dataset_path = Path(config.dataset_path)
    separation_dir = Path(config.separation_dir)
    
    # Separate dataset
    separation_fn = hydra.utils.instantiate(config.separation)
    separation_fn(output_dir=separation_dir, dataset_path=dataset_path)

    # Compute metrics
    results = evaluate_separations(separation_dir, dataset_path, 22050, eps=1e-8)

    # Store and show results
    results.to_csv(separation_dir/"metrics.csv")
    log.info(f'Results:\n{results[["bass","drums","guitar","piano"]].mean()}')


if __name__ == "__main__":
    main()