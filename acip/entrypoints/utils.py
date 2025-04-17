def setup_env(
    pythonpath: bool = True,
    dotenv: bool = True,
    cwd: bool = True,
    project_root_env_var: bool = True,
):
    """
    Utility function to make the setup of the project root robust and allows you to run the entrypoints from anywhere.
    This is a simple wrapper around the `rootutils.setup_root` function from https://github.com/ashleve/rootutils.
    To make this work properly, an empty file `.project-root` must be present in the root of the project.
    By Default, we also use a `.env` file to set environment variables.

    Notes: Execute this before Huggingface packages are imported to make sure that `HF_HOME` is already defined as
        an environment variable.
    """

    import rootutils

    rootutils.setup_root(
        __file__,
        indicator=".project-root",
        pythonpath=pythonpath,
        dotenv=dotenv,
        cwd=cwd,
        project_root_env_var=project_root_env_var,  # or set PROJECT_ROOT in .env
    )
