import polars as pl
import duckdb
from typing import Dict, Optional, Any, Generator
import numpy as np
from abc import ABC, abstractmethod

class Processor(ABC):
    """Abstract base class for batch processors"""
    
    @abstractmethod
    def process(self, batch: pl.DataFrame) -> pl.DataFrame:
        """Process a batch of data"""
        pass

class DataLoader:
    def __init__(self, subset_path: str):
        """
        Initialize the DataLoader with a path to a parquet file
        
        Args:
            subset_path: Path to the parquet file
        """
        # Read the schema first to identify numeric columns
        self.df = pl.scan_parquet(subset_path)
        self.schema = self.df.schema
        
        # Get total number of rows
        self.num_rows = self.df.select(pl.count()).collect().item()
        self.offset = 0
        
        # Calculate statistics for numeric columns (excluding array types)
        numeric_cols = [
            name for name, dtype in self.schema.items()
            if isinstance(dtype, (pl.Float64, pl.Int64)) and not str(dtype).endswith('[]')
        ]
        
        # Calculate means and variances
        stats = (
            self.df
            .select([
                pl.col(col).mean().alias(f"{col}_mean")
                for col in numeric_cols
            ] + [
                pl.col(col).var().alias(f"{col}_var")
                for col in numeric_cols
            ])
            .collect()
        )
        
        # Store statistics in dictionaries
        self.means = {
            col: stats[0][f"{col}_mean"] 
            for col in numeric_cols
        }
        self.variances = {
            col: stats[0][f"{col}_var"]
            for col in numeric_cols
        }
        
    def get_batch(
        self, 
        batch_size: int, 
        processor: Optional[Processor] = None
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Generate batches of data
        
        Args:
            batch_size: Number of rows per batch
            processor: Optional Processor object to transform the batch
            
        Yields:
            Processed or raw batch of data as a polars DataFrame
        """
        while self.offset < self.num_rows:
            # Read batch
            batch = (
                self.df
                .slice(self.offset, batch_size)
                .collect()
            )
            
            # Apply processor if provided
            if processor is not None:
                batch = processor.process(batch)
                
            self.offset += batch_size
            yield batch
            
    def reset(self):
        """Reset the offset to start from beginning"""
        self.offset = 0

# Example Processor implementations
class ColumnAdder(Processor):
    def __init__(self, column: str, value: Any):
        self.column = column
        self.value = value
    
    def process(self, batch: pl.DataFrame) -> pl.DataFrame:
        return batch.with_columns(
            pl.lit(self.value).alias(self.column)
        )

class ColumnReplacer(Processor):
    def __init__(self, 
                 target_column: str, 
                 source_columns: list[str], 
                 transform_func: callable):
        self.target_column = target_column
        self.source_columns = source_columns
        self.transform_func = transform_func
    
    def process(self, batch: pl.DataFrame) -> pl.DataFrame:
        # Get required columns as numpy arrays
        arrays = [
            batch[col].to_numpy() 
            for col in self.source_columns
        ]
        
        # Apply transformation
        new_values = self.transform_func(*arrays)
        
        # Replace column
        return batch.with_columns(
            pl.Series(name=self.target_column, values=new_values)
        )

if __name__ == '__main__':
    # Initialize loader
    loader = DataLoader("dataset/Train_Processed.parquet")

    # Print statistics
    print("Means:", loader.means)
    print("Variances:", loader.variances)

    # Example 1: Simple batch iteration
    for batch in loader.get_batch(batch_size=4000):
        print(f"Got batch of shape: {batch.shape}")

    # Example 2: Using a processor to add a column
    adder = ColumnAdder("new_col", 1.0)
    for batch in loader.get_batch(batch_size=32, processor=adder):
        print(f"Got processed batch with new column: {batch.columns}")

    # Example 3: Using a processor to replace a column
    def combine_sbp_dbp(sbp, dbp):
        return (sbp + dbp) / 2

    replacer = ColumnReplacer(
        target_column="MeanBP",
        source_columns=["SegSBP", "SegDBP"],
        transform_func=combine_sbp_dbp
    )

for batch in loader.get_batch(batch_size=32, processor=replacer):
    print(f"Got batch with mean BP: {batch['MeanBP'].mean()}")
