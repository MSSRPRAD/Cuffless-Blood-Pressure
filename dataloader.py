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

class ColumnAdder(Processor):
    def __init__(self, 
                 target_column: str, 
                 source_columns: list[str], 
                 transform_func: callable,
                 output_type: str = "numeric"):
        """
        Initialize ColumnAdder with target column, source columns and transform function
        
        Args:
            target_column: Name of the new column to be added
            source_columns: List of source column names to be used in transformation
            transform_func: Function that takes values from source columns and returns new value
            output_type: Type of output ("numeric" or "array")
        """
        self.target_column = target_column
        self.source_columns = source_columns
        self.transform_func = transform_func
        self.output_type = output_type
    
    def process(self, batch: pl.DataFrame) -> pl.DataFrame:
        """
        Process a batch by adding new column based on transformation of source columns
        
        Args:
            batch: Input DataFrame batch
            
        Returns:
            DataFrame with new column added
        """
        # Extract source columns
        source_data = [batch[col] for col in self.source_columns]
        
        # Apply transformation function
        new_column = self.transform_func(*source_data)
        
        if self.output_type == "array":
            # For array outputs, we need to handle each row separately
            # and store the result as a list of arrays
            return batch.with_columns([
                pl.Series(
                    name=self.target_column,
                    values=[arr for arr in new_column]
                )
            ])
        else:
            # For numeric outputs, we can directly create a series
            return batch.with_columns([
                pl.Series(name=self.target_column, values=new_column)
            ])


if __name__ == '__main__':
    # Initialize loader
    loader = DataLoader("dataset/Train_Processed.parquet")

    # Test 1: Average of SegSBP and SegDBP
    def calculate_avg(sbp, dbp):
        return (sbp + dbp) / 2

    avg_processor = ColumnAdder(
        target_column="AvgBP",
        source_columns=["SegSBP", "SegDBP"],
        transform_func=calculate_avg,
        output_type="numeric"
    )

    # Test 2: Combine PPG_F and ABP_F
    def combine_signals(ppg_f, abp_f):
        # Process each row individually
        combined = []
        for p, a in zip(ppg_f, abp_f):
            combined.append(np.vstack([p, a]).tolist())
        return combined

    signal_processor = ColumnAdder(
        target_column="CombinedSignals",
        source_columns=["PPG_F", "ABP_F"],
        transform_func=combine_signals,
        output_type="array"
    )

    # Process first batch with each processor
    for batch in loader.get_batch(batch_size=32, processor=avg_processor):
        print("\nTest 1 Results:")
        print("New column 'AvgBP' added:", "AvgBP" in batch.columns)
        print(batch.shape)
        print(batch.columns)
        print(batch.head())
        print("Sample values:")
        print(batch.select(["SegSBP", "SegDBP", "AvgBP"]).head(3))
        break

    # Reset loader
    loader.reset()

    for batch in loader.get_batch(batch_size=32, processor=signal_processor):
        print("\nTest 2 Results:")
        print("New column 'CombinedSignals' added:", "CombinedSignals" in batch.columns)
        print("Sample combined signal shape:", batch["CombinedSignals"][0].shape)
        print(batch.shape)
        print(batch.columns)
        print(batch.head())
        break
