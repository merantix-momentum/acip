"""
Helper entrypoint that loads and ACIP model and corresponding tokenizer and pushes it to the Hugging Face Hub.
"""
from acip.entrypoints.utils import setup_env

# Setup python paths and load environment variables from .env.
# Run this early to make sure that HF_HOME is defined.
setup_env()

import os
from logging import getLogger

import hydra
import torch
from accelerate import init_on_device
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from acip import PROJECT_ROOT
from acip.entrypoints import HYDRA_CONFIG_PATH
from acip.model.model_factory import ACIPModelFactory
from acip.model.tokenizer_factory import TokenizerFactory

logger = getLogger(__name__)


@hydra.main(
    config_path=HYDRA_CONFIG_PATH,
    config_name="entrypoints/acip_push_to_hub.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    # Resolve config to make print-outs easily readable.
    OmegaConf.resolve(cfg)

    # Print config
    if cfg.run.print_cfg:
        logger.info(f"Full hydra config:\n{OmegaConf.to_yaml(cfg)}")
    # Stop here if dry run
    if cfg.run.dry_run:
        return

    if cfg.acip.hub.push_model:
        # Build model
        model_factory: ACIPModelFactory = hydra.utils.instantiate(cfg.model.model_factory, _convert_="object")
        logger.info("Model factory created.")
        with init_on_device(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
            model = model_factory()
            if torch.cuda.is_available():
                model.cuda()
        logger.info("Model created.")

        # Build tokenizer
        tokenizer_factory: TokenizerFactory = hydra.utils.instantiate(cfg.model.tokenizer_factory, _convert_="object")
        logger.info("Tokenizer factory created.")
        tokenizer = tokenizer_factory()
        logger.info("Tokenizer created.")

        # Push model and tokenizer to hub
        model.push_to_hub(cfg.acip.hub.repo_id, commit_message="Add model")
        logger.info("Model pushed to hub.")
        tokenizer.push_to_hub(cfg.acip.hub.repo_id, commit_message="Add model")
        logger.info("Tokenizer pushed to hub.")
    else:
        logger.info("Skipping model push.")

    # Upload all additional files
    parent_dir = os.path.join(PROJECT_ROOT, "acip", "core")

    # Render temporary README and fill in templates
    with open(os.path.join(parent_dir, "README_template.md"), "r") as f:
        readme = f.read()
    readme = readme.replace("{{BASE_MODEL}}", cfg.model.base_model_name_or_path)
    readme = readme.replace("{{REPO_ID}}", cfg.acip.hub.repo_id)
    readme = readme.replace("{{LICENSE}}", cfg.acip.hub.license)
    readme = readme.replace("{{LICENSE_TEXT}}", cfg.acip.hub.license_text)
    with open(os.path.join(parent_dir, "README.md"), "w") as f:
        f.write(readme)

    upload_file_list = ["README.md", "requirements.txt"]
    api = HfApi()
    for file in upload_file_list:
        api.upload_file(
            path_or_fileobj=os.path.join(parent_dir, file),
            path_in_repo=file,
            repo_id=cfg.acip.hub.repo_id,
            commit_message=f"Add {file}",
        )
        logger.info(f"File {file} pushed to hub.")

    # Remove temporarily rendered README
    os.remove(os.path.join(parent_dir, "README.md"))


if __name__ == "__main__":
    main()
