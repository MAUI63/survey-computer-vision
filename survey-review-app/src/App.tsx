/** @jsxImportSource react */
import React, { useState, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, PolygonLayer, BitmapLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';
import { Map } from 'react-map-gl/maplibre';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import './App.css';

import Box from '@mui/material/Box';
import { ImageModal } from './ImageModal';

const INITIAL_VIEW_STATE = {
  longitude: 174.5,
  latitude: -37.2,
  zoom: 8.5,
};

const URL_BASE = 'http://localhost:8000';
// API key loaded from environment variable
const apiKey = import.meta.env.VITE_LINZ_BASEMAPS_API_KEY;

if (!apiKey) {
  console.warn('VITE_LINZ_BASEMAPS_API_KEY not found in environment variables');
}

interface FrameImage {
  path: string;
  bounds_lbrt: [number, number][];
  detections?: any[];
  [key: string]: any;
}

interface Frame {
  lon: number;
  lat: number;
  alt: number;
  images: { [key: string]: FrameImage };
  [key: string]: any;
}


function renderTooltip({ hoverInfo }) {
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
        <p>Alt: {object.alt.toFixed(1)} m</p>
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

  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<FrameImage | null>(null);

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


  const flatGpsFrameImages: FrameImage[] = useMemo(() => {
    if (!gpsFrames) return [];
    const items = gpsFrames.map(frame => Object.values(frame.images)).flat();
    items.forEach((item: FrameImage) => {
      item.detections = possibleDetectionsMap[item.path]?.detections;
    });
    return items;
  }, [gpsFrames, possibleDetectionsMap]);

  // Open modal with image at index
  const openImageModal = (image: FrameImage) => {
    setModalImage(image);
    setModalOpen(true);
  };

  const closeModal = () => setModalOpen(false);
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
    if (idx < cameraPhotos[camera].length - 1) {
      const nextImgIdx = cameraPhotos[camera][idx + 1];
      const nextImg = flatGpsFrameImages.find(img => img.path === nextImgIdx);
      if (nextImg) {
        setModalImage(nextImg);
      }
    }
  };

  const layers = [
    apiKey && new TileLayer({
      data: `https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api=${apiKey}`,
      visible: showBasemap,
      maxRequests: 50,
      minZoom: 0,
      maxZoom: 21,
      tileSize: 256,
      renderSubLayers: props => {
        const {
          bbox: { west, south, east, north }
        } = props.tile;
        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north]
        });
      },
    }),

    new ScatterplotLayer({
      id: 'frame-centers',
      data: gpsFrames,
      visible: showCentres,
      getPosition: d => [d.lon, d.lat, 0], // d.alt],
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
      data: flatGpsFrameImages,
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
      <DeckGL layers={layers} initialViewState={INITIAL_VIEW_STATE} controller={true} >
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
      </Box>


      {/* Fullscreen Modal for Image Viewing */}
      <ImageModal
        open={modalOpen}
        imageUrl={modalImage ? `${URL_BASE}/images/${modalImage.path}` : ''}
        title={modalImage ? modalImage.path + (['r28', 'l09'].includes(modalImage.path.split('/')[2].toLowerCase()) ? ' (rotated 180)' : '') : ''}
        detections={modalImage?.detections}
        rotate180={modalImage ? (['r28', 'l09'].includes(modalImage.path.split('/')[2].toLowerCase())) : false}
        onClose={closeModal}
        onPrev={gotoPrev}
        onNext={gotoNext}
      />
    </>
  );
}

export function renderToDOM(container: HTMLElement) {
  createRoot(container).render(<App />);
}
