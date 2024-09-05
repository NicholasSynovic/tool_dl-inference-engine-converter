from typing import Any

import click
from transformers import AutoModel


def loadModel(hfRepo: str) -> Any | None:
    model: Any
    try:
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=hfRepo,
            use_safetensors=True,
        )
    except OSError:
        model = None
        print(f"{hfRepo} does not contain a Safetensors model")

    return model


@click.command()
@click.option(
    "--hf-repo",
    "hfRepo",
    type=str,
    required=False,
    help="HuggingFace repository to download a Safetensor model from",
    default="sentence-transformers/all-mpnet-base-v2",
    show_default=True,
)
def main(hfRepo: str) -> None:
    model: Any | None = loadModel(hfRepo=hfRepo)

    if model is None:
        exit(1)


if __name__ == "__main__":
    main()
