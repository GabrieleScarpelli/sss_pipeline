from side_scan_sonar_pipeline import SideScanSonarPipeline
import argparse
import logging

# Line to run in bash:
#
# python sss_pipeline/main.py sss_pipeline/config.yaml


def main():
    parser = argparse.ArgumentParser(description="Run the Side Scan Sonar Processing Pipeline.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize and run the pipeline
    pipeline = SideScanSonarPipeline(args.config_path)
    logger.info("Pipeline initialized with configuration from %s", args.config_path)
    pipeline.run()
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()


