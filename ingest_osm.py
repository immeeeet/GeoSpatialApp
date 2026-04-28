"""
OSM PBF ingestion script — wrapper around osm2pgsql.

Imports OpenStreetMap data from a PBF file into PostGIS using the
'slim' schema, which creates:
  - planet_osm_point
  - planet_osm_line
  - planet_osm_polygon
  - planet_osm_roads

Usage:
    python ingest_osm.py                          # uses defaults
    python ingest_osm.py --pbf datasets/india.pbf # custom PBF path
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PBF = "datasets/india-latest.osm.pbf"

# Default PostGIS connection (override via env or CLI args)
DEFAULT_DB = "geospatial"
DEFAULT_USER = "postgres"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "5432"


def run_osm2pgsql(
    pbf_path: str,
    db: str = DEFAULT_DB,
    user: str = DEFAULT_USER,
    host: str = DEFAULT_HOST,
    port: str = DEFAULT_PORT,
    cache_mb: int = 2048,
    slim: bool = True,
) -> bool:
    """
    Run osm2pgsql to import a PBF file into PostGIS.
    Returns True on success, False on failure.
    """
    pbf = Path(pbf_path)
    if not pbf.exists():
        logger.error("PBF file not found: %s", pbf_path)
        logger.info("Download from: https://download.geofabrik.de/asia/india-latest.osm.pbf")
        return False

    cmd = [
        "osm2pgsql",
        "--create",             # Drop and recreate tables
        "--slim",               # Slim mode (required for updates)
        "--cache", str(cache_mb),
        "--number-processes", "4",
        "--host", host,
        "--port", port,
        "--database", db,
        "--username", user,
        "--latlong",            # Store coords in WGS84 (lat/lng)
        str(pbf),
    ]

    if not slim:
        cmd.remove("--slim")

    logger.info("Running: %s", " ".join(cmd))
    logger.info("This may take 30–60 minutes for India...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        logger.info("osm2pgsql import completed successfully!")
        return True
    except FileNotFoundError:
        logger.error(
            "osm2pgsql not found. Install it:\n"
            "  Ubuntu: sudo apt install osm2pgsql\n"
            "  macOS:  brew install osm2pgsql\n"
            "  Windows: download from https://osm2pgsql.org/doc/install.html"
        )
        return False
    except subprocess.CalledProcessError as exc:
        logger.error("osm2pgsql failed with return code %d", exc.returncode)
        return False


def main():
    parser = argparse.ArgumentParser(description="Import OSM PBF into PostGIS")
    parser.add_argument("--pbf", default=DEFAULT_PBF, help="Path to .osm.pbf file")
    parser.add_argument("--db", default=DEFAULT_DB, help="PostgreSQL database name")
    parser.add_argument("--user", default=DEFAULT_USER, help="PostgreSQL user")
    parser.add_argument("--host", default=DEFAULT_HOST, help="PostgreSQL host")
    parser.add_argument("--port", default=DEFAULT_PORT, help="PostgreSQL port")
    parser.add_argument("--cache", default=2048, type=int, help="Cache size in MB")
    args = parser.parse_args()

    success = run_osm2pgsql(
        pbf_path=args.pbf,
        db=args.db,
        user=args.user,
        host=args.host,
        port=args.port,
        cache_mb=args.cache,
    )

    if success:
        logger.info("Next step: Run the PostGIS index creation script:")
        logger.info("  psql -U %s -d %s -f postgres-init/01-init.sql", args.user, args.db)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
