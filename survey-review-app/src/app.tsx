/** @jsxImportSource react */
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, PolygonLayer, BitmapLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';
import { Map } from 'react-map-gl/maplibre';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import './app.css';

import Box from '@mui/material/Box';
import { ImageModal } from './ImageModal';

const INITIAL_VIEW_STATE = {
  longitude: 174.5,
  latitude: -37.2,
  zoom: 8.5,
};

// API key loaded from environment variable
const apiKey = import.meta.env.VITE_LINZ_BASEMAPS_API_KEY;
const dataURL = import.meta.env.VITE_DATA_API_URL;

if (!apiKey) {
  console.warn('VITE_LINZ_BASEMAPS_API_KEY not found in environment variables');
}

if (!dataURL) {
  console.warn('VITE_DATA_API_URL not found in environment variables, defaulting to localhost:8000');
}

const URL_BASE = dataURL || 'http://localhost:8000';

interface FrameImage {
  path: string;
  bounds_lbrt: [number, number][];
  iso: number;
  aperture: number;
  shutter_speed: number;
  agl: number;
  detections?: any[];
  [key: string]: any;
}

interface Frame {
  lon: number;
  lat: number;
  agl: number;
  images: { [key: string]: FrameImage };
  [key: string]: any;
}


function renderTooltip({ hoverInfo }: { hoverInfo: any }) {
  if (!hoverInfo) {
    return null;
  }
  const { object, x, y, layer } = hoverInfo;
  if (!object) {
    return null;
  }
  if (layer.id === 'frame-centers') {
    return (
      <div className="tooltip" style={{ left: x, top: y }}>
        <p>Lat: {object.lat.toFixed(5)}</p>
        <p>Lon: {object.lon.toFixed(5)}</p>
        <p>agl: {object.agl.toFixed(1)} m</p>
        <p>Time: {new Date(object.gps_time).toLocaleString()}</p>
        <p>Line: {object.line_id}</p>
        <p>Cameras: {Object.keys(object.images).join(', ')}</p>
      </div>
    );
  }
  else if (layer.id === 'frame-images') {
    return (
      <div className="tooltip" style={{ left: x, top: y }}>
        <p>Path: {object.path}</p>
        <p>Detections: {object.detections ? object.detections.length : 0}</p>
        <p>Bounds: {object.bounds_lbrt.map((b: [number, number]) => `[${b[0].toFixed(5)}, ${b[1].toFixed(5)}]`).join(', ')}</p>
        <p>ISO: {object.iso}</p>
        <p>Aperture: f/{object.aperture}</p>
        <p>Shutter Speed: 1/{1 / object.shutter_speed}</p>
      </div>
    );
  }
  return (
    <div className="tooltip" style={{ left: x, top: y }}>
      {JSON.stringify(object)}
    </div>
  );
}

export default function App({
  // mapStyle = 'https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json'
}) {
  const [hoverInfo, setHoverInfo] = useState<any | null>(null);
  const [cameraPhotos, setCameraPhotos] = useState<any>({});
  const [gpsFrames, setGPSFrames] = useState<Frame[]>([]);
  const [possibleDetections, setPossibleDetections] = useState<any[]>([]);
  const [showBasemap, setShowBasemap] = useState(true);
  const [showCentres, setShowCentres] = useState(true);
  const [showDetections, setShowDetections] = useState(false);
  const [showImgs, setShowImgs] = useState(true);
  const [lastClickedFrameCenter, setLastClickedFrameCenter] = useState<Frame | null>(null);

  // View state for auto-zoom functionality
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const deckRef = useRef<any>(null);

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<FrameImage | null>(null);

  // Filter states
  const [aglRange, setaglRange] = useState<[number, number]>([0, 1000]);
  const [isoRange, setIsoRange] = useState<[number, number]>([0, 2000]);
  const [shutterSpeedRange, setShutterSpeedRange] = useState<[number, number]>([100, 8000]); // 1/8000 to 1/100

  // Temporary states for slider dragging (for display purposes only)
  const [tempaglRange, setTempaglRange] = useState<[number, number]>([0, 1000]);
  const [tempIsoRange, setTempIsoRange] = useState<[number, number]>([0, 2000]);
  const [tempShutterSpeedRange, setTempShutterSpeedRange] = useState<[number, number]>([100, 8000]);

  // Sync temp states with actual filter states
  useEffect(() => {
    setTempaglRange(aglRange);
  }, [aglRange]);

  useEffect(() => {
    setTempIsoRange(isoRange);
  }, [isoRange]);

  useEffect(() => {
    setTempShutterSpeedRange(shutterSpeedRange);
  }, [shutterSpeedRange]);

  useEffect(() => {
    setCameraPhotos([]);
    fetch(`${URL_BASE}/files/camera_photos.json?${+new Date()}`)
      .then(response => response.json())
      .then(data => setCameraPhotos(data));
  }, []);

  useEffect(() => {
    setGPSFrames([]);
    fetch(`${URL_BASE}/files/frames.json?${+new Date()}`)
      .then(response => response.json())
      .then(data => {
        // Add idx:
        const framesWithIdx = data.sort((a: Frame, b: Frame) => new Date(a.gps_time).getTime() - new Date(b.gps_time).getTime()).map((frame: Frame, idx: number) => ({ ...frame, idx }));
        setGPSFrames(framesWithIdx);

        // Auto-zoom to GPS data area
        if (framesWithIdx.length > 0) {
          const lats = framesWithIdx.map((f: Frame) => f.lat);
          const lons = framesWithIdx.map((f: Frame) => f.lon);
          const minLat = Math.min(...lats);
          const maxLat = Math.max(...lats);
          const minLon = Math.min(...lons);
          const maxLon = Math.max(...lons);

          const centerLat = (minLat + maxLat) / 2;
          const centerLon = (minLon + maxLon) / 2;

          // Calculate zoom level based on bounds
          const latDiff = maxLat - minLat;
          const lonDiff = maxLon - minLon;
          const maxDiff = Math.max(latDiff, lonDiff);

          // Rough zoom calculation (adjust as needed)
          let zoom = 10;
          if (maxDiff > 1) zoom = 7;
          else if (maxDiff > 0.5) zoom = 8;
          else if (maxDiff > 0.1) zoom = 9;
          else if (maxDiff > 0.05) zoom = 10;
          else if (maxDiff > 0.01) zoom = 11;
          else zoom = 12;

          const newViewState = {
            longitude: centerLon,
            latitude: centerLat,
            zoom: zoom,
            transitionDuration: 2000,
            transitionInterpolator: undefined
          };

          setViewState(newViewState);
        }
      });
  }, []);

  useEffect(() => {
    setPossibleDetections([]);
    fetch(`${URL_BASE}/files/possible_detections.json?${+new Date()}`)
      .then(response => response.json())
      .then(data => {
        setPossibleDetections(data);
      });
  }, []);


  const possibleDetectionsMap = useMemo(() => {
    const map: { [key: string]: any } = {};
    possibleDetections.forEach(d => {
      map[d.img_path] = d;
    });
    return map;
  }, [possibleDetections]);

  // Helper function to check if value is at range boundary (meaning include all beyond)
  const isAtMinBoundary = (value: number, min: number): boolean => value === min;
  const isAtMaxBoundary = (value: number, max: number): boolean => value === max;


  const flatGpsFrameImages: FrameImage[] = useMemo(() => {
    if (!gpsFrames) return [];
    let items = gpsFrames.map(frame => {
      return Object.values(frame.images).map((img => ({ ...img, agl: frame.agl })));
    }).flat();

    items.forEach((item: FrameImage) => {
      item.detections = possibleDetectionsMap[item.path]?.detections;
    });
    return items;
  }, [gpsFrames, possibleDetectionsMap, aglRange, isoRange, shutterSpeedRange]);

  const filteredFlatGpsFrameImages = useMemo(() => {
    return flatGpsFrameImages.filter(frame => {
      // Apply agl filter to frame level
      const aglOk = isAtMinBoundary(aglRange[0], 0) || frame.agl >= aglRange[0];
      const aglOkMax = isAtMaxBoundary(aglRange[1], 1000) || frame.agl <= aglRange[1];
      // Apply ISO filter
      const isoOk = isAtMinBoundary(isoRange[0], 0) || frame.iso >= isoRange[0];
      const isoOkMax = isAtMaxBoundary(isoRange[1], 2000) || frame.iso <= isoRange[1];
      const shutterOk = isAtMinBoundary(shutterSpeedRange[0], 100) || 1 / frame.shutter_speed >= shutterSpeedRange[0];
      const shutterOkMax = isAtMaxBoundary(shutterSpeedRange[1], 8000) || 1 / frame.shutter_speed <= shutterSpeedRange[1];
      return aglOk && aglOkMax && isoOk && isoOkMax && shutterOk && shutterOkMax;
    });
  }, [flatGpsFrameImages, aglRange, isoRange, shutterSpeedRange]);

  const filteredGPSFrames = useMemo(() => {
    if (!gpsFrames) return [];
    return gpsFrames.filter(frame => {
      // Apply agl filter to frame level
      const aglOk = isAtMinBoundary(aglRange[0], 0) || frame.agl >= aglRange[0];
      const aglOkMax = isAtMaxBoundary(aglRange[1], 1000) || frame.agl <= aglRange[1];
      return aglOk && aglOkMax;
    });
  }, [gpsFrames, aglRange]);

  // Open modal with image at index
  const openImageModal = (image: FrameImage) => {
    setModalImage(image);
    setModalOpen(true);
  };

  const closeModal = () => setModalOpen(false);

  // Camera navigation functions
  const cameraOrder = ['l28', 'l09', 'r09', 'r28'];

  const gotoLeftCamera = () => {
    if (!modalImage) return;
    // Find the current frame that contains this image
    const currentFrame = gpsFrames.find(frame =>
      Object.values(frame.images).some(img => img.path === modalImage.path)
    );
    if (!currentFrame) return;

    // Find current camera
    const currentCamera = modalImage.path.split('/')[2].toLowerCase();
    const currentIndex = cameraOrder.indexOf(currentCamera);
    if (currentIndex === -1) return;

    // Get previous camera (wrap around)
    const prevIndex = currentIndex === 0 ? cameraOrder.length - 1 : currentIndex - 1;
    const prevCamera = cameraOrder[prevIndex];

    // Find image for that camera in the same frame
    if (currentFrame.images[prevCamera]) {
      setModalImage(currentFrame.images[prevCamera]);
    }
  };

  const gotoRightCamera = () => {
    if (!modalImage) return;
    // Find the current frame that contains this image
    const currentFrame = gpsFrames.find(frame =>
      Object.values(frame.images).some(img => img.path === modalImage.path)
    );
    if (!currentFrame) return;

    // Find current camera
    const currentCamera = modalImage.path.split('/')[2].toLowerCase();
    const currentIndex = cameraOrder.indexOf(currentCamera);
    if (currentIndex === -1) return;

    // Get next camera (wrap around)
    const nextIndex = (currentIndex + 1) % cameraOrder.length;
    const nextCamera = cameraOrder[nextIndex];

    // Find image for that camera in the same frame
    if (currentFrame.images[nextCamera]) {
      setModalImage(currentFrame.images[nextCamera]);
    }
  };

  const gotoPrev = () => {
    if (!modalImage) return;
    if (modalImage.path === '') return;
    const camera = modalImage.path.split('/')[2];
    const idx = cameraPhotos[camera].findIndex((p: string) => p === modalImage.path);
    if (idx > 0) {
      const nextImgIdx = cameraPhotos[camera][idx - 1];
      const nextImg = flatGpsFrameImages.find(img => img.path === nextImgIdx);
      if (nextImg) {
        setModalImage(nextImg);
      }
    }
  };
  const gotoNext = () => {
    if (!modalImage) return;
    if (modalImage.path === '') return;
    const camera = modalImage.path.split('/')[2];
    const idx = cameraPhotos[camera].findIndex((p: string) => p === modalImage.path);
    console.log(modalImage.path);
    console.log(idx);
    if (idx < cameraPhotos[camera].length - 1) {
      const nextImgIdx = cameraPhotos[camera][idx + 1];
      const nextImg = flatGpsFrameImages.find(img => img.path === nextImgIdx);
      if (nextImg) {
        setModalImage(nextImg);
      }
    }
  };

  const layers = [
    ...(apiKey ? [new TileLayer({
      data: `https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api=${apiKey}`,
      visible: showBasemap,
      maxRequests: 50,
      minZoom: 0,
      maxZoom: 21,
      tileSize: 256,
      renderSubLayers: (props: any) => {
        const {
          bbox: { west, south, east, north }
        } = props.tile;
        return new BitmapLayer(props, {
          data: undefined,
          image: props.data,
          bounds: [west, south, east, north]
        });
      },
    })] : []),

    new ScatterplotLayer({
      id: 'frame-centers',
      data: filteredGPSFrames,
      visible: showCentres,
      getPosition: d => [d.lon, d.lat, 0], // d.agl],
      filled: true,
      stroked: false,
      getFillColor: d => {
        if (lastClickedFrameCenter && d.idx === lastClickedFrameCenter.idx) {
          return [0, 0, 0, 255]; // Highlight last clicked in black
        }
        const t = new Date(d.gps_time);
        const seconds = t.getHours() * 3600 + t.getMinutes() * 60 + t.getSeconds();
        const ratio = (seconds - 8 * 3600) / (12 * 3600); // seconds in a day
        const r = Math.floor(255 * ratio);
        const g = Math.floor(255 * (1 - ratio));
        return [r, g, 0, 200];
      },
      getRadius: 20,
      radiusMinPixels: 3,
      pickable: true,
      onHover: setHoverInfo,
      updateTriggers: {
        getFillColor: [lastClickedFrameCenter]
      },
      onClick: (o: { object?: Frame }) => {
        if (o.object) {
          const b = o.object;
          console.log(b);
          if (lastClickedFrameCenter) {
            // Print info about distance and time difference
            const a = lastClickedFrameCenter;
            const R = 6371e3; // metres
            const φ1 = a.lat * Math.PI / 180; // φ, λ in radians
            const φ2 = b.lat * Math.PI / 180;
            const Δφ = (b.lat - a.lat) * Math.PI / 180;
            const Δλ = (b.lon - a.lon) * Math.PI / 180;
            const haversine = (x: number) => Math.sin(x / 2) * Math.sin(x / 2);
            const c = 2 * Math.atan2(Math.sqrt(haversine(Δφ) + Math.cos(φ1) * Math.cos(φ2) * haversine(Δλ)), Math.sqrt(1 - (haversine(Δφ) + Math.cos(φ1) * Math.cos(φ2) * haversine(Δλ)))); // angular
            const d = R * c; // in metres
            const timeDiff = (new Date(b.gps_time).getTime() - new Date(a.gps_time).getTime()) / 1000;
            const o = {
              'from_idx': a.idx,
              'from_time': a.gps_time,
              'from_lat': a.lat.toFixed(6),
              'from_lon': a.lon.toFixed(6),
              'to_idx': b.idx,
              'to_time': b.gps_time,
              'to_lat': b.lat.toFixed(6),
              'to_lon': b.lon.toFixed(6),
              'num_frames': b.idx - a.idx,
              'length_m': d.toFixed(0),
              'duration_s': timeDiff.toFixed(0),
              'average_speed_kph': (d / timeDiff * 3.6).toFixed(0),
            }
            if (o.num_frames > 0) {
              console.log(JSON.stringify(o));
            }
          }
          setLastClickedFrameCenter(b);
        }
      }

    }),

    new PolygonLayer({
      id: 'frame-images',
      data: filteredFlatGpsFrameImages,
      visible: showImgs,
      getPolygon: (d: FrameImage) => d.bounds_lbrt.map((b: [number, number]) => [b[0], b[1], d.detections ? 10 : 1]),
      stroked: true,
      getFillColor: d => {
        if (d.detections) {
          return [255, 0, 0, 100]; // Red if detection exists
        }
        return [0, 0, 255, 50];
      },
      lineWidthMinPixels: 0,
      getLineWidth: 5,
      pickable: true,
      onHover: setHoverInfo,
      onClick: (o: { object?: FrameImage }) => {
        if (o.object) {
          console.log(o.object);
          openImageModal(o.object);
        }
      }
    }),


    new PolygonLayer({
      id: 'frame-detections',
      data: flatGpsFrameImages.filter(img => img.detections && img.detections.length > 0),
      visible: showDetections,
      getPolygon: (d: FrameImage) => d.bounds_lbrt.map((b: [number, number]) => [b[0], b[1], d.detections ? 10 : 1]),
      getFillColor: [0, 0, 255, 255],
      getLineColor: [0, 0, 255, 255],
      getLineWidth: 5,
      lineWidthMinPixels: 5,
      pickable: true,
      onHover: setHoverInfo,
      onClick: (o: { object?: FrameImage }) => {
        if (o.object) {
          openImageModal(o.object);
        }
      }
    }),
  ];

  // modalImage && console.log(modalImage);

  return (
    <>
      <DeckGL
        ref={deckRef}
        layers={layers}
        viewState={viewState}
        onViewStateChange={({ viewState: newViewState }) => setViewState(newViewState as any)}
        controller={true}
      >
        <Map reuseMaps />
      </DeckGL>
      {!modalOpen && renderTooltip({ hoverInfo })}
      <Box
        sx={{
          position: 'absolute',
          top: 20,
          left: 20,
          width: 300,
          padding: '20px',
          backgroundColor: "rgba(255,255,255,0.2)"
        }}
      >
        <p>For reviewing the surveys. Use the slider to pick the time to see images - we don't have enough memory (?) to show everything in one go.</p>
        <FormGroup>
          <FormControlLabel control={<Switch checked={showBasemap} onChange={e => setShowBasemap(e.target.checked)} />} label="Basemap" />
          <FormControlLabel control={<Switch checked={showCentres} onChange={e => setShowCentres(e.target.checked)} />} label="Frame Centers" />
          <FormControlLabel control={<Switch checked={showImgs} onChange={e => setShowImgs(e.target.checked)} />} label="Images" />
          <FormControlLabel control={<Switch checked={showDetections} onChange={e => setShowDetections(e.target.checked)} />} label="Images with Detections" />
        </FormGroup>

        {/* Filter Controls */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>Filters</Typography>

          {/* agl Filter */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
              agl: {tempaglRange[0] === 0 ? '0' : tempaglRange[0]} - {tempaglRange[1] === 1000 ? '1000+' : tempaglRange[1]} m
            </Typography>
            <Slider
              value={tempaglRange}
              onChange={(_, newValue) => setTempaglRange(newValue as [number, number])}
              onChangeCommitted={(_, newValue) => setaglRange(newValue as [number, number])}
              valueLabelDisplay="auto"
              min={0}
              max={1000}
              step={10}
              marks={[
                { value: 0, label: '0' },
                { value: 500, label: '500' },
                { value: 1000, label: '1000+' }
              ]}
            />
          </Box>

          {/* ISO Filter */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
              ISO: {tempIsoRange[0] === 0 ? '0' : tempIsoRange[0]} - {tempIsoRange[1] === 2000 ? '2000+' : tempIsoRange[1]}
            </Typography>
            <Slider
              value={tempIsoRange}
              onChange={(_, newValue) => setTempIsoRange(newValue as [number, number])}
              onChangeCommitted={(_, newValue) => setIsoRange(newValue as [number, number])}
              valueLabelDisplay="auto"
              min={0}
              max={2000}
              step={50}
              marks={[
                { value: 0, label: '0' },
                { value: 800, label: '800' },
                { value: 1600, label: '1600' },
                { value: 2000, label: '2000+' }
              ]}
            />
          </Box>

          {/* Shutter Speed Filter */}
          <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
              Shutter Speed: 1/{tempShutterSpeedRange[0] === 100 ? '<100' : tempShutterSpeedRange[0]} - 1/{tempShutterSpeedRange[1] === 8000 ? '8000+' : tempShutterSpeedRange[1]}
            </Typography>
            <Slider
              value={tempShutterSpeedRange}
              onChange={(_, newValue) => setTempShutterSpeedRange(newValue as [number, number])}
              onChangeCommitted={(_, newValue) => setShutterSpeedRange(newValue as [number, number])}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `1/${value}`}
              min={100}
              max={8000}
              step={100}
              marks={[
                { value: 1000, label: '1/1000' },
                { value: 2500, label: '1/2500' },
                { value: 5000, label: '1/5000' },
                { value: 8000, label: '1/8000+' }
              ]}
            />
          </Box>
        </Box>
      </Box>


      {/* Fullscreen Modal for Image Viewing */}
      <ImageModal
        open={modalOpen}
        imageUrl={modalImage ? `${URL_BASE}/images/${modalImage.path}` : ''}
        title={modalImage ? modalImage.path + (['r28', 'l09'].includes(modalImage.path.split('/')[2].toLowerCase()) ? ' (rotated 180)' : '') : ''}
        detections={modalImage?.detections}
        iso={modalImage?.iso}
        aperture={modalImage?.aperture}
        shutterSpeed={modalImage?.shutter_speed}
        agl={modalImage?.agl}
        rotate180={modalImage ? (['r28', 'l09'].includes(modalImage.path.split('/')[2].toLowerCase())) : false}
        onClose={closeModal}
        onPrev={gotoPrev}
        onNext={gotoNext}
        onLeftCamera={gotoLeftCamera}
        onRightCamera={gotoRightCamera}
      />
    </>
  );
}

export function renderToDOM(container: HTMLElement) {
  createRoot(container).render(<App />);
}
