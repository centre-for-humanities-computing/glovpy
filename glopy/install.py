import shutil
import subprocess
from pathlib import Path


class InstallationError(Exception):
    pass


def is_installed() -> bool:
    installation_path = Path.home().joinpath(".glopy", "build")
    for executable in ["vocab_count", "cooccur", "shuffle", "glove"]:
        if not installation_path.joinpath(executable).exists():
            return False
    else:
        return True


def install_glove():
    glopy_path = Path.home().joinpath(".glopy")
    glopy_path.mkdir(exist_ok=True)
    glopy_abs = str(glopy_path.absolute())
    git_path = shutil.which("git")
    make_path = shutil.which("make")
    subprocess.run(
        [
            git_path or "git",
            "clone",
            "https://github.com/stanfordnlp/GloVe.git",
            glopy_abs,
        ]
    )
    subprocess.run(make_path or "make", cwd=glopy_path)
    if not is_installed():
        raise InstallationError(
            "Installing Glove failed on your system. "
            "Please check if you have an appropriate compiler."
        )
