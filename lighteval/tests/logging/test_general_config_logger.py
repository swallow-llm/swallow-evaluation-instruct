import os
import subprocess

from lighteval.logging import info_loggers
from lighteval.logging.info_loggers import GeneralConfigLogger


def test_lighteval_sha_is_head_of_enclosing_repo():
    """lighteval_sha must be the HEAD of the git repository that contains the installed lighteval sources."""
    src_dir = os.path.dirname(os.path.abspath(info_loggers.__file__))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src_dir, capture_output=True, text=True, check=True)
    assert GeneralConfigLogger().lighteval_sha == head.stdout.strip()
