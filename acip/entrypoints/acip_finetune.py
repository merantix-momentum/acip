"""
Executes the ACIP finetune entrypoint, which is typically a follow-up of the ACIP compress entrypoint.
Please see `README.md` for a more detailed user guide.
"""

from acip.entrypoints.utils import setup_env

# Setup python paths and load environment variables from .env.
# Run this early to make sure that HF_HOME is defined.
setup_env()

import hydra

from acip.entrypoints import HYDRA_CONFIG_PATH
from acip.entrypoints.acip_entrypoint import run_acip_entrypoint

if __name__ == "__main__":
    main = hydra.main(
        config_path=HYDRA_CONFIG_PATH,
        config_name="entrypoints/acip_finetune.yaml",
        version_base="1.3",
    )(run_acip_entrypoint)
    main()
