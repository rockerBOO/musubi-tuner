"""Regression test for #921: cache_latents' encode_datasets progress bar should show
a real percentage via tqdm(total=...), driven by BaseDataset.get_total_image_count().

This locks in the PR author's open question: whether get_total_image_count() (the
raw indexable-datasource length) is actually equal to the number of items yielded
across all batches from retrieve_latent_cache_batches() -- i.e. the total tqdm
would be iterating toward. Built from a real multi-dataset TOML config, mirroring
how cache_latents.py's main() loads datasets in practice.
"""

import argparse
import textwrap

from PIL import Image

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO


def _make_image_dir(dir_path, num_images, size=(64, 64)):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        Image.new("RGB", size, color=(i * 10 % 256, 0, 0)).save(dir_path / f"img_{i:03d}.jpg")
        (dir_path / f"img_{i:03d}.txt").write_text("a caption")


def _load_dataset_group(dataset_config_path):
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(str(dataset_config_path))
    blueprint = blueprint_generator.generate(user_config, argparse.Namespace(), architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    return config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)


def test_get_total_image_count_matches_items_yielded_by_retrieve_latent_cache_batches(tmp_path):
    # Mirrors a real dataset_config.toml: multiple [[datasets]] blocks under one [general].
    dataset_a = tmp_path / "set_a"
    dataset_b = tmp_path / "set_b"
    _make_image_dir(dataset_a, num_images=3)
    _make_image_dir(dataset_b, num_images=2)

    config_path = tmp_path / "dataset_config.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            [general]
            caption_extension = '.txt'
            enable_bucket = true
            resolution = 64

            [[datasets]]
            image_directory = '{dataset_a}'
            cache_directory = '{tmp_path / "cache_a"}'

            [[datasets]]
            image_directory = '{dataset_b}'
            cache_directory = '{tmp_path / "cache_b"}'
            """
        )
    )

    dataset_group = _load_dataset_group(config_path)
    datasets = dataset_group.datasets
    assert len(datasets) == 2

    for dataset, expected_count in zip(datasets, [3, 2]):
        total = dataset.get_total_image_count()
        assert total == expected_count

        items_yielded = sum(len(batch) for _, batch in dataset.retrieve_latent_cache_batches(num_workers=1))
        assert items_yielded == total


def test_get_total_image_count_returns_none_for_non_indexable_datasource(tmp_path):
    dataset_dir = tmp_path / "set_a"
    _make_image_dir(dataset_dir, num_images=2)

    config_path = tmp_path / "dataset_config.toml"
    config_path.write_text(
        textwrap.dedent(
            f"""\
            [general]
            caption_extension = '.txt'
            enable_bucket = true
            resolution = 64

            [[datasets]]
            image_directory = '{dataset_dir}'
            cache_directory = '{tmp_path / "cache"}'
            """
        )
    )

    dataset = _load_dataset_group(config_path).datasets[0]
    dataset.datasource.is_indexable = lambda: False

    assert dataset.get_total_image_count() is None
