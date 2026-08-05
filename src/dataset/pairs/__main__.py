import argparse
import datetime
import os

import dataset.dataset_read as dr
from dataset.dataset_filtration import DatasetFiltration
from dataset.pairs.create_custom_bin import FacePairGenerator

from utils.config_loader import Config
from utils.log_utils import setup_hybrid_logger
from utils.paths import ProjectPathResolver


def main():
    parser = argparse.ArgumentParser(description="Creation Face Pairs")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path (e.g. configs/config.face_pair.yaml)")
    parser.add_argument("--ground_truth_type", type=str, default="gt_labat", choices=["gt_labat", "gt_semantic"], help="Type of Ground Truth")
    parser.add_argument("--filter_enable", action="store_true", help="Enable filter pre-coded for dataset, look the code for more information",
    )
    args = parser.parse_args()

    gt_type = args.ground_truth_type
    if gt_type not in ["gt_labat", "gt_semantic"]:
        raise SystemExit("Use: gt_labat or gt_semantic")

    config = Config(path=args.config)
    path_resolver = ProjectPathResolver(config)
    df = dr.load_dataset(path_resolver['dataset.csv_path'])

    log_name = config["bins_pairs.results.log.name"]
    # Get current date and time
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H%M%S")
    log_file_path = path_resolver["bins_pairs.results.log_path"] / (log_name + "_" + formatted_now + ".log")
    logger = setup_hybrid_logger(
        name=log_name,
        level="INFO",
        log_file=log_file_path.path,
        stream_target="auto",
        notebook_friendly=False,
        clear_handlers=True,
    )

    logger.info(f"Args: {args}")
    logger.info("Starting Face Pairs Generation")
    logger.info(f"Length of full dataframe: {len(df)}")


    if args.filter_enable:
        # This part is only with filters, datetime hours of the day.
        df_filtration = DatasetFiltration(df)
        filters_cfg = config.get("bins_pairs.filters", {}) or {}
        hour_range = filters_cfg.get("hour_range", None)
        custom_filter = None
        if hour_range is not None:
            h_min, h_max = hour_range
            custom_filter = lambda df_: df_['date_time'].dt.hour.between(h_min, h_max)

        column_filters = filters_cfg.get("column_filters", None)
        range_filters = filters_cfg.get("range_filters", None)
        value_filters = filters_cfg.get("value_filters", None)
        boolean_filters = filters_cfg.get("boolean_filters", None)
        drop_na_columns = filters_cfg.get("drop_na_columns", None)
        keep_columns = filters_cfg.get("keep_columns", None)

        logger.info(f"Applying filters to dataset: {filters_cfg}")

        df = df_filtration.filter(
            column_filters=column_filters,
            range_filters=range_filters,
            value_filters=value_filters,
            boolean_filters=boolean_filters,
            drop_na_columns=drop_na_columns,
            keep_columns=keep_columns,
            custom_filter=custom_filter,
            inplace=False  # mantém self._df intacto; retorna um novo DF
        )

        logger.info(f"Length of filtered dataframe: {len(df)}")

    bins_pairs_parameters = config['bins_pairs.parameters']

    generator = FacePairGenerator(logger, **bins_pairs_parameters)

    result_root_path = path_resolver['bins_pairs.results.root_path']
    os.makedirs(result_root_path.path, exist_ok=True)

    generator.generate(df, result_root_path, gt_type, filtered=args.filter_enable)


if __name__ == "__main__":
    main()