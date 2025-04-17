"""
This subpackage contains the available entrypoints for the ACIP project.

We current support `acip_compress.py` and `acip_finetune.py`, which are both based on
`acip_entrypoint.ACIPEntrypoint`. Each entrypoint is based on hydra config management.
The configs are rooted in "config" (see also `HYDRA_CONFIG_PATH`). The config folder structure mirrors the
subpackage structure of `acip`, which is also reflected in `acip_entrypoint.ACIPEntrypointConf`.

Please see `README.md` for a more detailed user guide.
"""

import os

from acip import PROJECT_ROOT

# Path to the hydra config directory
HYDRA_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config")
