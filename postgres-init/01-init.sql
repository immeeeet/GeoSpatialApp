-- PostGIS initialization script
-- Runs automatically when the postgres container starts for the first time

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS hstore;

-- Create spatial indexes on OSM tables (created by osm2pgsql)
-- These indexes dramatically speed up ST_DWithin queries

-- Note: These will fail silently if the tables don't exist yet
-- (i.e., before osm2pgsql import). That's fine — run them again after import.

DO $$
BEGIN
    -- Index on planet_osm_line (roads)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'planet_osm_line') THEN
        CREATE INDEX IF NOT EXISTS idx_osm_line_highway ON planet_osm_line(highway) WHERE highway IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_line_way_geog ON planet_osm_line USING GIST ((way::geography));
        RAISE NOTICE 'Created indexes on planet_osm_line';
    END IF;

    -- Index on planet_osm_polygon (buildings, land use)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'planet_osm_polygon') THEN
        CREATE INDEX IF NOT EXISTS idx_osm_poly_landuse ON planet_osm_polygon(landuse) WHERE landuse IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_poly_building ON planet_osm_polygon(building) WHERE building IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_poly_way_geog ON planet_osm_polygon USING GIST ((way::geography));
        RAISE NOTICE 'Created indexes on planet_osm_polygon';
    END IF;

    -- Index on planet_osm_point (POIs, amenities)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'planet_osm_point') THEN
        CREATE INDEX IF NOT EXISTS idx_osm_point_amenity ON planet_osm_point(amenity) WHERE amenity IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_point_shop ON planet_osm_point(shop) WHERE shop IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_point_leisure ON planet_osm_point(leisure) WHERE leisure IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_osm_point_way_geog ON planet_osm_point USING GIST ((way::geography));
        RAISE NOTICE 'Created indexes on planet_osm_point';
    END IF;
END $$;
