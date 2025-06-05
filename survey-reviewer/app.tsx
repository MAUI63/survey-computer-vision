import React, { useState, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';
import { BitmapLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/maplibre';
// import { MapViewState } from '@deck.gl/core';
// import { WebMercatorViewport } from '@deck.gl/core';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';

import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';

const INITIAL_VIEW_STATE = {
    longitude: 174.5,
    latitude: -37.5,
    zoom: 9,
};

const URL_BASE = 'http://127.0.0.1:8001/data';

function renderTooltip({ hoverInfo }) {
    if (!hoverInfo) {
        return null;
    }
    const { object, x, y } = hoverInfo;
    if (!object) {
        return null;
    }
    console.log(object);
    return (
        <div className="tooltip" style={{ left: x, top: y }}>
            {JSON.stringify(object)}
        </div>
    );
}

function parsePackedImages(bites) {
    const out = {};
    const n = bites.length;
    const dv = new DataView(bites.buffer);
    let offset = 0;
    while (true) {
        if (offset >= n - 1) {
            break;
        }
        const nameLen = dv.getUint32(offset);
        offset += 4;
        const bitesLen = dv.getUint32(offset);
        offset += 4;
        if (offset + nameLen + bitesLen > n) {
            break;
        }
        const name = new TextDecoder('utf-8').decode(bites.subarray(offset, offset + nameLen));
        // check name endswith .jpg
        if (!name.endsWith('.jpg')) {
            console.log("Name does not end with .jpg: ", name);
            break;
        }
        offset += nameLen;
        const imgBites = new Uint8Array(bites.buffer, offset, bitesLen);
        offset += bitesLen;
        const imgData = new Blob([imgBites], { type: 'image/jpeg' });
        const imgUrl = URL.createObjectURL(imgData);
        out[name] = imgUrl;
    }
    return out;
}

export default function App({
    mapStyle = 'https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json'
}) {
    const [hoverInfo, setHoverInfo] = useState({});
    const [imgs, setImgs] = useState([]);
    const [imgLookup, setImgLookup] = useState({});
    const [showBasemap, setShowBasemap] = useState(true);
    const [showCentres, setShowCentres] = useState(true);
    const [showImgs, setShowImgs] = useState(true);
    const flightPpnDisplayMax = 0.1;
    const flightPpnDisplayMin = 0.05;
    const [flightPpn, setFlightPpn] = useState([0, flightPpnDisplayMax]);
    // const [bounds, setBounds] = useState(null);
    // const [viewState, setViewState] = useState<MapViewState>({
    //     longitude: INITIAL_VIEW_STATE.longitude,
    //     latitude: INITIAL_VIEW_STATE.latitude,
    //     zoom: INITIAL_VIEW_STATE.zoom,
    // });

    useEffect(() => {
        setImgs([]);
        fetch(`${URL_BASE}/imgs.json?${+new Date()}`)
            .then(response => response.json())
            .then(data => setImgs(data));
    }, []);

    useEffect(() => {
        setImgLookup({});
        fetch(`${URL_BASE}/imgs.bin?${+new Date()}`)
            .then(response => response.arrayBuffer())
            .then(bites => {
                const parsedImages = parsePackedImages(new Uint8Array(bites));
                setImgLookup(parsedImages);
            });
    }, []);

    console.log("Found images: ", imgs.length);
    console.log("Found images lookup: ", Object.keys(imgLookup).length);

    const handleSliderChange = (event: Event, newValue: number | number[], activeThumb: number) => {
        if (newValue[1] - newValue[0] > flightPpnDisplayMax) {
            if (activeThumb === 0) {
                setFlightPpn([newValue[0], Math.min(newValue[0] + flightPpnDisplayMax, 1)]);
            } else {
                setFlightPpn([Math.max(0, newValue[1] - flightPpnDisplayMax), newValue[1]]);
            }
        } else if (newValue[1] - newValue[0] < flightPpnDisplayMin) {
            if (activeThumb === 0) {
                setFlightPpn([newValue[0], Math.min(newValue[0] + flightPpnDisplayMin, 1)]);
            } else {
                setFlightPpn([Math.max(0, newValue[1] - flightPpnDisplayMin), newValue[1]]);
            }
        } else {
            setFlightPpn(newValue);
        }
    };

    const frameLayers = useMemo(() => {
        if (imgs && showImgs) {
            // const bounds = viewState.bounds;
            // const nw = bounds[0];
            // const se = bounds[1];
            let filtered = imgs.filter(img => imgLookup.hasOwnProperty(img.thumbnail_name));
            filtered = filtered.filter(img => img.flight_ppn >= flightPpn[0] && img.flight_ppn <= flightPpn[1]);
            // filtered = filtered.filter(img => {
            //     const lon = img.location.lon;
            //     const lat = img.location.lat;
            //     return lon >= nw[0] && lon <= se[0] && lat <= nw[1] && lat >= se[1];
            // })
            // if (filtered.length > 1000) {
            //     filtered = filtered.slice(0, 100);
            // }
            const frameLayers = filtered.map((img, i) => {
                // raise the bounds up:
                const height = img.flight_ppn * 100;
                const bounds = img.bounds_lbrt.map(b => [b[0], b[1], height]);
                return new BitmapLayer({
                    id: `bitmap-layer-${i}`,
                    data: [img],
                    // image: `${URL_BASE}/thumbnails/${img.thumbnail_name}`,
                    image: imgLookup[img.thumbnail_name],
                    bounds: bounds,
                    opacity: 0.5,
                    // pickable: true,
                    // onHover: setHoverInfo,
                    // onClick: (o) => {
                    //     console.log(o);
                    //     console.log(img);
                    // }
                });
            });
            return frameLayers
        }

    }, [imgs, imgLookup, flightPpn]);

    const layers = [
        new TileLayer({
            data: 'https://tiles-cdn.koordinates.com/services;key=c345590fda674252971b30cfe32b65bc/tiles/v4/layer=115053/EPSG:3857/{z}/{x}/{y}.png',
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
            data: imgs,
            visible: showCentres,
            getPosition: d => [d.location.lon, d.location.lat, d.location.alt],
            filled: true,
            stroked: false,
            getFillColor: [0, 0, 255, 100],
            getRadius: 10,
            radiusMinPixels: 2,
            pickable: true,
            onHover: setHoverInfo
        }),


    ];
    if (frameLayers) {
        layers.push(...frameLayers);
    }

    return (
        <>
            <DeckGL
                // viewState={viewState}
                // onViewStateChange={e => {
                //     setViewState(e.viewState);
                //     const viewport = new WebMercatorViewport(e.viewState);
                //     const nw = viewport.unproject([0, 0]);
                //     const se = viewport.unproject([viewport.width, viewport.height]);
                //     setBounds([nw, se]);
                // }}
                layers={layers}
                initialViewState={INITIAL_VIEW_STATE}
                controller={true}
            >
                <Map reuseMaps mapStyle={mapStyle} />
            </DeckGL>
            {renderTooltip({ hoverInfo })}
            <Box
                sx={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
                    width: 300,
                    backgroundColor: "rgba(255,255,255,0.2)"
                }}
            >
                <p>For reviewing the surveys. Use the slider to pick the time to see images - we don't have enough memory (?) to show everything in one go.</p>
                <FormGroup>
                    <FormControlLabel control={<Switch checked={showBasemap} onChange={e => setShowBasemap(e.target.checked)} />} label="Basemap" />
                    <FormControlLabel control={<Switch checked={showCentres} onChange={e => setShowCentres(e.target.checked)} />} label="Frame Centers" />
                    <FormControlLabel control={<Switch checked={showImgs} onChange={e => setShowImgs(e.target.checked)} />} label="Images" />
                </FormGroup>
                <Slider
                    valueLabelDisplay="auto"
                    value={flightPpn}
                    min={0}
                    max={1}
                    step={0.001}
                    onChange={handleSliderChange}
                />
            </Box>
        </>
    );
}

export function renderToDOM(container) {
    createRoot(container).render(<App />);
}
