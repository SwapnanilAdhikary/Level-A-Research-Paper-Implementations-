"""Script to generate the benchmark dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multicollude.data.generator import DatasetGenerator, create_default_config


def main():
    """Generate the benchmark dataset."""
    # Create configuration
    config = create_default_config()

    # Override output directory
    config.output_dir = str(Path(__file__).parent.parent / "data")

    # Create generator
    generator = DatasetGenerator(config)

    # Generate dataset
    output_path = generator.generate()

    print(f"\nDataset generation complete!")
    print(f"Output: {output_path}")
    print(f"\nFiles generated:")
    for file in Path(config.output_dir).glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
