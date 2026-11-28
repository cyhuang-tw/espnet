"""Tests for the AMI configs that decode and pack checkpoints."""

import ami_sot_paths
from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_and_merge_config


def _load(name, role):
    """Load a recipe config the way run.py does, without resolving it."""
    config = load_and_merge_config(
        ami_sot_paths.RECIPE / "conf" / name,
        config_name=f"{role}.yaml",
        default_package=ami_sot_paths.TEMPLATE_PACKAGE,
        resolve=False,
    )
    return OmegaConf.to_container(config)


def test_the_released_training_config_changes_only_the_exp_tag():
    """training_whisper_sot.yaml is training.yaml with its own exp_tag."""
    released = _load("training_whisper_sot.yaml", "training")
    default = _load("training.yaml", "training")

    assert released.pop("exp_tag") == "whisper_small_sot"
    default.pop("exp_tag")
    assert released == default


def test_the_released_inference_config_decodes_config_yaml_and_model_pth():
    """inference_whisper_sot.yaml reads the two released files from exp_dir.

    The inference directory also changes, because its name comes from the name
    of the config file. All other settings are those of inference.yaml.
    """
    released = _load("inference_whisper_sot.yaml", "inference")
    default = _load("inference.yaml", "inference")

    assert released["model"].pop("s2t_train_config") == "${exp_dir}/config.yaml"
    assert released["model"].pop("s2t_model_file") == "${exp_dir}/model.pth"
    assert released.pop("inference_dir") == "${exp_dir}/inference_whisper_sot"
    for key in ("s2t_train_config", "s2t_model_file"):
        default["model"].pop(key)
    default.pop("inference_dir")
    assert released == default


def test_the_released_publication_config_adds_only_a_description():
    """publication_whisper_sot.yaml is publication.yaml with a model description.

    The exp_tag of a publication config comes from the name of the file, and
    run.py replaces it with the exp_tag of the training config.
    """
    released = _load("publication_whisper_sot.yaml", "publication")
    default = _load("publication.yaml", "publication")

    context = released["pack_model"].pop("readme_context")
    assert "was not used to train this checkpoint" in context["description"]
    released.pop("exp_tag")
    default.pop("exp_tag")
    assert released == default


def test_the_recipe_decodes_with_the_sot_speech2text_and_runner():
    """The AMI inference config sets the SOT model, the SOT runner and no batching.

    The template uses espnet2's Speech2Text and the plain runner, so the recipe
    has to set the SOT versions of both.
    """
    config = _load("inference.yaml", "inference")

    assert config["model"]["_target_"] == (
        "espnet3.systems.esp2_s2t.src.sot.s2t_inference.Speech2Text"
    )
    assert config["runner"]["_target_"] == (
        "espnet3.systems.esp2_s2t.src.sot.inference_runner.SotInferenceRunner"
    )
    assert config["batch_size"] is None
