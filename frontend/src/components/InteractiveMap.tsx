import React, { useState, useCallback } from 'react';
import Map, { MapEvent } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// Ensure you replace this with your actual Mapbox access token, 
// ideally from an environment variable: import.meta.env.VITE_MAPBOX_TOKEN
const MAPBOX_TOKEN = "pk.eyJ1IjoiZHVtbXlfdXNlciIsImEiOiJjbXhxYXozejAwMDByMmxsYnpzYzZsZ2h4In0.dummy_token_123456";

const InteractiveMap: React.FC = () => {
  const [viewState, setViewState] = useState({
    longitude: 78.9629,
    latitude: 20.5937,
    zoom: 4
  });

  const handleMapClick = useCallback(async (event: MapEvent<MouseEvent>) => {
    // The event.lngLat object contains the coordinates of the clicked point
    const lng = event.lngLat.lng;
    const lat = event.lngLat.lat;

    console.log(`Map clicked at: Lat ${lat}, Lon ${lng}`);

    try {
      // Assuming Vite proxy or CORS allows /api/v1 prefix mapping to the FastAPI backend
      const response = await fetch('/api/v1/analyze-site', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          latitude: lat,
          longitude: lng,
          use_case: 'retail' // Default use case for this boilerplate
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Site Analysis Response from Backend:', data);
    } catch (error) {
      console.error('Error fetching analysis data:', error);
    }
  }, []);

  return (
    <div className="w-full h-full min-h-[500px]">
      <Map
        {...viewState}
        onMove={evt => setViewState(evt.viewState)}
        onClick={handleMapClick}
        mapStyle="mapbox://styles/mapbox/streets-v11"
        mapboxAccessToken={MAPBOX_TOKEN}
        style={{ width: '100%', height: '100%' }}
        cursor="crosshair"
      />
    </div>
  );
};

export default InteractiveMap;
