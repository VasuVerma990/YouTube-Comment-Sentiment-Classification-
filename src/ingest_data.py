import logging
import yaml
from typing import Tuple
import pandas as pd
from zenml import step

with open(r"C:\Users\Vasu\Desktop\youtube\config.yaml", "r") as file:
    config = yaml.safe_load(file)
enable_tracking = config["experiment_tracking"]["enable_tracking"]


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(r"C:\Users\Vasu\Desktop\youtube\Data\YoutubeCommentsDataSet.csv")
        return df


@step(enable_cache=enable_tracking)
def ingest_data() -> Tuple[pd.DataFrame, str]:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        col_name = "Comment"
        return df, col_name
    except Exception as e:
        logging.error(e)
        raise e
