import React from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

interface ImageModalProps {
    open: boolean;
    imageUrl: string;
    title: string;
    detections: any[] | undefined;
    rotate180: boolean;
    onClose: () => void;
    onPrev: () => void;
    onNext: () => void;
}

export function ImageModal({ open, imageUrl, detections, onClose, onPrev, onNext, rotate180, title }: ImageModalProps) {
    const [loaded, setLoaded] = React.useState(false);
    const [showDetections, setShowDetections] = React.useState(true);
    React.useEffect(() => {
        setLoaded(false);
    }, [imageUrl]);
    if (!open) return null;
    // Extract image name from path
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            background: 'rgba(0,0,0,0.95)',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
        }}>
            {/* Title bar for image name */}
            <div style={{
                position: 'absolute',
                top: 20,
                left: '50%',
                transform: 'translateX(-50%)',
                color: '#fff',
                background: 'rgba(0,0,0,0.7)',
                padding: '6px 18px',
                borderRadius: 8,
                fontSize: 18,
                zIndex: 10001,
                maxWidth: '80vw',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
            }}>{title}</div>
            <button style={{ position: 'absolute', top: 20, right: 20, zIndex: 10000 }} onClick={onClose}>Close</button>
            {/* Show/hide detections button at top left */}
            <button style={{ position: 'absolute', top: 20, left: 20, zIndex: 10000 }} onClick={() => setShowDetections(v => !v)}>
                {showDetections ? 'Hide' : 'Show'} Detections
            </button>
            {/* Next button at top center, below close */}
            <button style={{ position: 'absolute', top: 80, left: '50%', transform: 'translateX(-50%)', zIndex: 10000 }} onClick={onNext}>{'⬆️'}</button>
            {/* Prev button at bottom center */}
            <button style={{ position: 'absolute', bottom: 40, left: '50%', transform: 'translateX(-50%)', zIndex: 10000 }} onClick={onPrev}>{'⬇️'}</button>
            {/* Loader spinner outside zoom/pan */}
            {!loaded && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 10001,
                }}>
                    <div style={{
                        border: '8px solid #f3f3f3',
                        borderTop: '8px solid #3498db',
                        borderRadius: '50%',
                        width: 60,
                        height: 60,
                        animation: 'spin 1s linear infinite',
                    }} />
                    <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
                </div>
            )}
            <div style={{
                width: '90vw',
                height: '90vh',
                minWidth: 300,
                minHeight: 300,
                maxWidth: '100vw',
                maxHeight: '100vh',
                background: '#222',
                position: 'relative',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
            }}>
                <TransformWrapper
                    initialScale={.7}
                    minScale={0.7}
                    maxScale={20}
                    wheel={{ smoothStep: 0.01 }}
                    doubleClick={{ disabled: false }}
                    panning={{ velocityDisabled: true }}
                >
                    <TransformComponent>
                        <ImageWithOverlay src={imageUrl} detections={showDetections ? detections : []} setLoaded={setLoaded} rotate180={rotate180} />
                    </TransformComponent>
                </TransformWrapper>
            </div>
        </div>
    );
}

function ImageWithOverlay({ src, detections, setLoaded, rotate180 }: { src: string; detections: any[] | undefined; setLoaded: (v: boolean) => void, rotate180: boolean }) {
    const [loaded, setLocalLoaded] = React.useState(false);
    React.useEffect(() => {
        setLocalLoaded(false);
        setLoaded(false);
    }, [src, setLoaded]);

    return (
        <div style={{
            position: 'relative', width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            transform: rotate180 ? 'rotate(180deg)' : 'none',
            border: '2px solid white', // Border so we know we're at the edge when we pan vs offscreen

        }}>
            <img
                src={src}
                alt="modal"
                onLoad={() => { setLoaded(true); setLocalLoaded(true); }}
                style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                    display: 'block',
                    margin: '0 auto',
                    position: 'relative',
                    visibility: loaded ? 'visible' : 'hidden',
                }}
            />
            {/* Example bounding box overlay at xywh=[0.5,0.5,0.01,0.01] */}
            {loaded && detections && detections.map((det, i) => (
                <div
                    key={i}
                    style={{
                        position: 'absolute',
                        border: `1px solid #00FF00`,
                        left: `${det.ltwh[0] * 100}%`,
                        top: `${det.ltwh[1] * 100}%`,
                        width: `${det.ltwh[2] * 100}%`,
                        height: `${det.ltwh[3] * 100}%`,
                        pointerEvents: 'none',
                    }}
                />
            ))}
            {/* TODO: Render more bounding boxes here using relative coordinates */}
        </div>
    );
}
