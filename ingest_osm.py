import os
import logging
from itertools import islice
from dotenv import load_dotenv

import fiona
import geopandas as gpd
from sqlalchemy import create_engine, text

# -------------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Example DB connection string format: postgresql://user:password@host:port/dbname
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/geodb")
ENGINE = create_engine(DB_URL)

# Mapping of shapefile names to their corresponding PostGIS table names
SHAPEFILES_TO_PROCESS = {
    "gis_osm_roads_free_1.shp": "osm_roads",
    "gis_osm_buildings_a_free_1.shp": "osm_buildings",
    "gis_osm_pois_free_1.shp": "osm_pois"
}

# -------------------------------------------------------------------
# Core ETL Function
# -------------------------------------------------------------------
def ingest_shapefile_in_chunks(filepath: str, table_name: str, engine, chunk_size: int = 50000):
    """
    Ingests a shapefile into a PostGIS table.
    Uses Fiona to stream the shapefile in memory-manageable chunks,
    converting to GeoPandas DataFrames and inserting via SQLAlchemy/GeoAlchemy2.
    """
    if not os.path.exists(filepath):
        logger.error(f"Shapefile not found: {filepath}. Skipping.")
        return

    logger.info(f"Starting ingestion process for {filepath} -> table '{table_name}'")
    
    first_chunk = True
    
    # fiona.open() allows streaming file reading without loading everything into memory
    with fiona.open(filepath) as src:
        crs = src.crs
        total_records = len(src)
        logger.info(f"Detected {total_records} features. CRS: {crs}")
        
        records_processed = 0
        iterator = iter(src)
        
        while True:
            # Extract the next batch of features using islice
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break  # End of file
            
            # Convert chunk to a GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(chunk, crs=crs)
            
            # Standardize CRS: Ensure EVERYTHING is projected to EPSG:4326 before inserting
            if gdf.crs is None:
                logger.warning("No CRS found in shapefile! Assuming EPSG:4326.")
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                logger.info(f"Reprojecting chunk from {gdf.crs} to EPSG:4326")
                gdf = gdf.to_crs(epsg=4326)
            
            # Write chunk to PostGIS
            if first_chunk:
                # Replace existing table on the first chunk
                # Geopandas handles creating the table structure including the geometry column
                gdf.to_postgis(
                    name=table_name,
                    con=engine,
                    if_exists="replace",
                    index=False,
                    chunksize=10000  # Database insert batch size
                )
                first_chunk = False
            else:
                # Append subsequent chunks
                gdf.to_postgis(
                    name=table_name,
                    con=engine,
                    if_exists="append",
                    index=False,
                    chunksize=10000
                )
                
            records_processed += len(chunk)
            logger.info(f"Progress [{table_name}]: {records_processed} / {total_records} features inserted.")
            
    # Enforce spatial index creation post-ingestion
    # (to_postgis creates one automatically if if_exists='replace', but this guarantees it natively on the column)
    logger.info(f"Enforcing GIST spatial index on '{table_name}' geometry column...")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geom ON {table_name} USING GIST (geometry);"))

    logger.info(f"Successfully completed ingestion for {table_name}.\n")

# -------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------
def main():
    # Directory where user dumped the unzipped Geofabrik `.shp` files
    data_dir = os.getenv("SHAPEFILE_DIR", "./data")
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory '{data_dir}' does not exist! Please set SHAPEFILE_DIR or paths.")
    
    # Process the required layers
    for filename, table_name in SHAPEFILES_TO_PROCESS.items():
        filepath = os.path.join(data_dir, filename)
        ingest_shapefile_in_chunks(filepath, table_name, ENGINE)
        
    logger.info("All specified OSM shapefiles have been successfully processed and indexed!")

if __name__ == "__main__":
    main()
