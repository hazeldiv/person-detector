"use client";

import { useRef, useState, useCallback, useEffect } from "react";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";
const TARGET_W = 640;
const TARGET_H = 480;
const SEND_FPS = 15;
const JPEG_QUALITY = 0.7;
/** Max frames to buffer (covers ~2s of round-trip at 15fps) */
const FRAME_BUFFER_MAX = 60;
/** Max polyline points a user can add */
const MAX_POLY_POINTS = 32;

/* ── Types ── */
interface BBox {
  id: number;
  x: number;
  y: number;
  w: number;
  h: number;
  weight: number;
}

interface Point {
  x: number;
  y: number;
}

type DrawStep = "idle" | "drawing" | "pick-side" | "active";

/* ── Letterbox helpers ── */
function computeLetterbox(srcW: number, srcH: number) {
  const scale = Math.min(TARGET_W / srcW, TARGET_H / srcH);
  const nw = Math.round(srcW * scale);
  const nh = Math.round(srcH * scale);
  const padX = Math.floor((TARGET_W - nw) / 2);
  const padY = Math.floor((TARGET_H - nh) / 2);
  return { scale, nw, nh, padX, padY };
}

/** Map a point from original camera space → 640×480 letterbox space */
function toLetterboxPt(
  px: number,
  py: number,
  lb: ReturnType<typeof computeLetterbox>
) {
  return { x: px * lb.scale + lb.padX, y: py * lb.scale + lb.padY };
}

/** Map a bbox from 640×480 letterbox space → original camera space */
function fromLetterbox(
  bx: number,
  by: number,
  bw: number,
  bh: number,
  lb: ReturnType<typeof computeLetterbox>
) {
  return {
    x: (bx - lb.padX) / lb.scale,
    y: (by - lb.padY) / lb.scale,
    w: bw / lb.scale,
    h: bh / lb.scale,
  };
}

/**
 * Extend a single endpoint to the nearest canvas border.
 * `p`     = the endpoint to extend
 * `from`  = the adjacent interior point (defines the ray direction away from it)
 * Returns the border-snapped version of `p`.
 */
function extendEndpoint(p: Point, from: Point, w: number, h: number): Point {
  const margin = 5;
  // Already on the border — just clamp
  if (
    p.x <= margin ||
    p.x >= w - margin ||
    p.y <= margin ||
    p.y >= h - margin
  ) {
    return { x: Math.max(0, Math.min(w, p.x)), y: Math.max(0, Math.min(h, p.y)) };
  }

  // Direction: from `from` toward `p`, then continue to border
  const dx = p.x - from.x;
  const dy = p.y - from.y;
  const candidates: Point[] = [];

  const tryBorder = (t: number, bx: number, by: number) => {
    if (t > 0 && bx >= 0 && bx <= w && by >= 0 && by <= h)
      candidates.push({ x: bx, y: by });
  };

  if (Math.abs(dx) > 1e-9) {
    let t = -p.x / dx;
    tryBorder(t, 0, p.y + t * dy);
    t = (w - p.x) / dx;
    tryBorder(t, w, p.y + t * dy);
  }
  if (Math.abs(dy) > 1e-9) {
    let t = -p.y / dy;
    tryBorder(t, p.x + t * dx, 0);
    t = (h - p.y) / dy;
    tryBorder(t, p.x + t * dx, h);
  }

  if (candidates.length === 0) {
    // Fallback: snap to nearest border
    const dists = [
      { pt: { x: 0, y: p.y }, d: p.x },
      { pt: { x: w, y: p.y }, d: w - p.x },
      { pt: { x: p.x, y: 0 }, d: p.y },
      { pt: { x: p.x, y: h }, d: h - p.y },
    ];
    dists.sort((a, b) => a.d - b.d);
    return dists[0].pt;
  }

  // Closest border intersection
  let best = candidates[0];
  let bestD = Math.hypot(best.x - p.x, best.y - p.y);
  for (let i = 1; i < candidates.length; i++) {
    const d = Math.hypot(candidates[i].x - p.x, candidates[i].y - p.y);
    if (d < bestD) { best = candidates[i]; bestD = d; }
  }
  return best;
}

/**
 * Apply extendEndpoint to first and last points of a polyline.
 * The intermediate points are kept exactly as drawn.
 */
function extendPolylineToBorder(pts: Point[], w: number, h: number): Point[] {
  if (pts.length < 2) return pts;
  const result = [...pts];
  result[0] = extendEndpoint(pts[0], pts[1], w, h);
  result[result.length - 1] = extendEndpoint(
    pts[pts.length - 1],
    pts[pts.length - 2],
    w,
    h
  );
  return result;
}

/**
 * Horizontal ray-cast parity: fires a ray from (px,py) rightward and counts
 * how many polyline segments it crosses. Returns 0 or 1.
 * Used to determine which side of the polyline a point is on.
 */
function raycastParity(px: number, py: number, pts: Point[]): number {
  let count = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const x1 = pts[i].x, y1 = pts[i].y;
    const x2 = pts[i + 1].x, y2 = pts[i + 1].y;
    if ((y1 <= py && py < y2) || (y2 <= py && py < y1)) {
      const t = (py - y1) / (y2 - y1);
      const ix = x1 + t * (x2 - x1);
      if (ix > px) count++;
    }
  }
  return count % 2;
}

/**
 * Snap a point to the nearest border of the w x h rectangle and return
 * its distance along the perimeter (starting at top-left going clockwise).
 */
function getBorderParam(p: Point, w: number, h: number): number {
  const x = Math.max(0, Math.min(w, p.x));
  const y = Math.max(0, Math.min(h, p.y));
  const dLeft = x;
  const dRight = w - x;
  const dTop = y;
  const dBottom = h - y;
  const minDist = Math.min(dLeft, dRight, dTop, dBottom);
  if (minDist === dTop) {
    return x;
  } else if (minDist === dRight) {
    return w + y;
  } else if (minDist === dBottom) {
    return w + h + (w - x);
  } else {
    return 2 * w + h + (h - y);
  }
}

/**
 * Get all corner points of the w x h rectangle that lie in the clockwise
 * path from p1 to p2 on the boundary.
 */
function getClockwiseBorderPath(p1: Point, p2: Point, w: number, h: number): Point[] {
  const d1 = getBorderParam(p1, w, h);
  const d2 = getBorderParam(p2, w, h);
  const total = 2 * (w + h);

  const corners = [
    { pt: { x: 0, y: 0 }, d: 0 },
    { pt: { x: w, y: 0 }, d: w },
    { pt: { x: w, y: h }, d: w + h },
    { pt: { x: 0, y: h }, d: 2 * w + h },
  ];

  let targetD = d2;
  if (targetD < d1) {
    targetD += total;
  }

  const candidates: { pt: Point; cd: number }[] = [];
  for (const c of corners) {
    // Check cycle 1 (0 to total)
    if (c.d > d1 && c.d < targetD) {
      candidates.push({ pt: c.pt, cd: c.d });
    }
    // Check cycle 2 (total to 2*total)
    const dCycle2 = c.d + total;
    if (dCycle2 > d1 && dCycle2 < targetD) {
      candidates.push({ pt: c.pt, cd: dCycle2 });
    }
  }

  candidates.sort((a, b) => a.cd - b.cd);
  return candidates.map((c) => c.pt);
}

/**
 * Calculate the arithmetic average centroid of a polygon.
 */
function getPolygonCentroid(pts: Point[]): Point {
  if (pts.length === 0) return { x: 0, y: 0 };
  let x = 0, y = 0;
  for (const p of pts) {
    x += p.x;
    y += p.y;
  }
  return { x: x / pts.length, y: y / pts.length };
}

/* ══════════════════════════════════════ */
/*              Main Component            */
/* ══════════════════════════════════════ */
export default function Home() {
  /* ── Refs ── */
  const videoRef = useRef<HTMLVideoElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const sendCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animFrameRef = useRef<number>(0);
  const cameraResRef = useRef({ w: 0, h: 0 });
  const letterboxRef = useRef<ReturnType<typeof computeLetterbox> | null>(null);

  // Frame synchronization
  const frameIdRef = useRef<number>(0);
  const frameBufferRef = useRef<Map<number, ImageBitmap>>(new Map());
  const pendingDrawRef = useRef<{ frameId: number; boxes: BBox[]; fps: number } | null>(null);
  const displayFrameRef = useRef<ImageBitmap | null>(null);
  const displayBoxesRef = useRef<BBox[]>([]);

  // Polyline drawing (live refs so drawOutput can read without stale closure)
  const drawnPointsRef = useRef<Point[]>([]);
  const polylineRef = useRef<Point[] | null>(null);
  const drawStepRef = useRef<DrawStep>("idle");
  const mousePosRef = useRef<Point | null>(null);

  // Double-click detection
  const lastClickTimeRef = useRef<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ── State (for React rendering only) ── */
  const [inputSource, setInputSource] = useState<"camera" | "video">("camera");
  const [isStreaming, setIsStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [serverFps, setServerFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [cameraRes, setCameraRes] = useState("");
  const [drawStep, setDrawStep] = useState<DrawStep>("idle");
  const [pointCount, setPointCount] = useState(0);
  const [sideLabel, setSideLabel] = useState<"A" | "B" | null>(null);

  /* ── Connect WebSocket ── */
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => { setWsConnected(true); console.log("[WS] Connected"); };
    ws.onclose = () => { setWsConnected(false); console.log("[WS] Disconnected"); };
    ws.onerror = () => setWsConnected(false);

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "detections") {
          pendingDrawRef.current = { frameId: data.frameId, boxes: data.boxes, fps: data.fps };
          setDetectionCount(data.boxes.length);
          setServerFps(data.fps);
        }
      } catch { /* ignore */ }
    };

    wsRef.current = ws;
  }, []);

  /* ── Draw output canvas (runs at camera fps via rAF) ── */
  const drawOutput = useCallback(() => {
    const canvas = outputCanvasRef.current;
    const lb = letterboxRef.current;
    if (!canvas || !lb) {
      animFrameRef.current = requestAnimationFrame(drawOutput);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      animFrameRef.current = requestAnimationFrame(drawOutput);
      return;
    }

    const camW = cameraResRef.current.w;
    const camH = cameraResRef.current.h;

    // Consume pending server detection — swap in the matching buffered frame
    const pending = pendingDrawRef.current;
    if (pending) {
      pendingDrawRef.current = null;
      const bitmap = frameBufferRef.current.get(pending.frameId);
      if (bitmap) {
        if (displayFrameRef.current && displayFrameRef.current !== bitmap)
          displayFrameRef.current.close();
        displayFrameRef.current = bitmap;
        displayBoxesRef.current = pending.boxes;
        for (const [id, bmp] of frameBufferRef.current) {
          if (id <= pending.frameId) {
            if (bmp !== bitmap) bmp.close();
            frameBufferRef.current.delete(id);
          }
        }
      }
    }

    // Draw server-synchronized frame (fallback to live video before first response)
    if (displayFrameRef.current) {
      ctx.drawImage(displayFrameRef.current, 0, 0, camW, camH);
    } else {
      const video = videoRef.current;
      if (video) ctx.drawImage(video, 0, 0, camW, camH);
    }

    // ── Draw polyline ──
    const step = drawStepRef.current;
    const pts = drawnPointsRef.current;
    const poly = polylineRef.current;

    // Active / pick-side: draw the final extended polyline
    const renderPoly = poly ?? (pts.length >= 2 ? null : null);
    if ((step === "active" || step === "pick-side") && poly && poly.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([10, 5]);
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(poly[0].x, poly[0].y);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x, poly[i].y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Endpoint dots
      for (const ep of [poly[0], poly[poly.length - 1]]) {
        ctx.beginPath();
        ctx.arc(ep.x, ep.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#fbbf24";
        ctx.fill();
      }

      // Detection side arrow at midpoint of the polyline
      if (step === "active" && sideLabel) {
        const mid = Math.floor(poly.length / 2);
        const mx = (poly[mid - 1]?.x ?? poly[0].x + poly[mid].x) / 2;
        const my = (poly[mid - 1]?.y ?? poly[0].y + poly[mid].y) / 2;
        const ldx = poly[mid].x - poly[mid - 1 >= 0 ? mid - 1 : 0].x;
        const ldy = poly[mid].y - poly[mid - 1 >= 0 ? mid - 1 : 0].y;
        const len = Math.hypot(ldx, ldy) || 1;
        const nx = -ldy / len, ny = ldx / len;
        const arrowLen = 36;
        const ax = mx + nx * arrowLen, ay = my + ny * arrowLen;

        ctx.strokeStyle = "#4f8cff";
        ctx.fillStyle = "rgba(79,140,255,0.25)";
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(mx, my); ctx.lineTo(ax, ay); ctx.stroke();
        ctx.beginPath(); ctx.arc(ax, ay, 6, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#4f8cff";
        ctx.font = "bold 12px Inter, sans-serif";
        ctx.fillText("DETECT", ax + 10, ay + 4);
      }
      ctx.restore();
    }

    // Drawing in progress: draw segments so far + rubber-band to mouse
    if (step === "drawing" && pts.length > 0) {
      ctx.save();
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.setLineDash([6, 3]);
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);

      // Rubber-band segment to mouse cursor
      const mouse = mousePosRef.current;
      if (mouse) ctx.lineTo(mouse.x, mouse.y);

      ctx.stroke();
      ctx.setLineDash([]);

      // Vertex dots
      for (const pt of pts) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#fbbf24";
        ctx.fill();
      }
      ctx.restore();
    }

    // Pick-side mode: show pulsing hint overlays and hover highlight
    if (step === "pick-side" && poly && poly.length >= 2) {
      ctx.save();

      // Build the two region paths
      const reg1 = [
        ...poly,
        ...getClockwiseBorderPath(poly[poly.length - 1], poly[0], camW, camH)
      ];
      const reg2 = [
        ...[...poly].reverse(),
        ...getClockwiseBorderPath(poly[0], poly[poly.length - 1], camW, camH)
      ];

      // Centroids
      const cent1 = getPolygonCentroid(reg1);
      const cent2 = getPolygonCentroid(reg2);
      const parity1 = raycastParity(cent1.x, cent1.y, poly);
      const label1 = parity1 === 0 ? "Side A" : "Side B";
      const label2 = parity1 === 0 ? "Side B" : "Side A";

      // Check hover
      const mouse = mousePosRef.current;
      let hoveredRegion: 1 | 2 | null = null;

      if (mouse) {
        // Test Region 1
        ctx.beginPath();
        if (reg1.length > 0) {
          ctx.moveTo(reg1[0].x, reg1[0].y);
          for (let i = 1; i < reg1.length; i++) ctx.lineTo(reg1[i].x, reg1[i].y);
        }
        ctx.closePath();
        if (ctx.isPointInPath(mouse.x, mouse.y)) {
          hoveredRegion = 1;
        } else {
          // Test Region 2
          ctx.beginPath();
          if (reg2.length > 0) {
            ctx.moveTo(reg2[0].x, reg2[0].y);
            for (let i = 1; i < reg2.length; i++) ctx.lineTo(reg2[i].x, reg2[i].y);
          }
          ctx.closePath();
          if (ctx.isPointInPath(mouse.x, mouse.y)) {
            hoveredRegion = 2;
          }
        }
      }

      // Draw Region 1 overlay
      ctx.beginPath();
      if (reg1.length > 0) {
        ctx.moveTo(reg1[0].x, reg1[0].y);
        for (let i = 1; i < reg1.length; i++) ctx.lineTo(reg1[i].x, reg1[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = hoveredRegion === 1 ? "rgba(0, 0, 0, 0.55)" : "rgba(0, 0, 0, 0.2)";
      ctx.fill();

      // Draw Region 2 overlay
      ctx.beginPath();
      if (reg2.length > 0) {
        ctx.moveTo(reg2[0].x, reg2[0].y);
        for (let i = 1; i < reg2.length; i++) ctx.lineTo(reg2[i].x, reg2[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = hoveredRegion === 2 ? "rgba(0, 0, 0, 0.55)" : "rgba(0, 0, 0, 0.2)";
      ctx.fill();

      // Draw Side Labels at Centroids
      const drawLabel = (cent: Point, label: string, isHovered: boolean) => {
        ctx.save();
        ctx.font = "bold 13px Inter, sans-serif";
        const text = isHovered ? `${label} (Monitor)` : label;
        const tw = ctx.measureText(text).width;
        
        // Background card
        ctx.fillStyle = isHovered ? "rgba(79, 140, 255, 0.9)" : "rgba(0, 0, 0, 0.65)";
        ctx.beginPath();
        const px = cent.x - tw / 2 - 10;
        const py = cent.y - 12;
        const pw = tw + 20;
        const ph = 24;
        const r = 4; // corner radius
        ctx.moveTo(px + r, py);
        ctx.lineTo(px + pw - r, py);
        ctx.quadraticCurveTo(px + pw, py, px + pw, py + r);
        ctx.lineTo(px + pw, py + ph - r);
        ctx.quadraticCurveTo(px + pw, py + ph, px + pw - r, py + ph);
        ctx.lineTo(px + r, py + ph);
        ctx.quadraticCurveTo(px, py + ph, px, py + ph - r);
        ctx.lineTo(px, py + r);
        ctx.quadraticCurveTo(px, py, px + r, py);
        ctx.closePath();
        ctx.fill();
        if (isHovered) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Text
        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, cent.x, cent.y);
        ctx.restore();
      };

      drawLabel(cent1, label1, hoveredRegion === 1);
      drawLabel(cent2, label2, hoveredRegion === 2);

      ctx.restore();

      // Label at bottom
      ctx.save();
      ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
      ctx.fillRect(0, camH - 40, camW, 40);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = "bold 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Click on the side you want to DETECT", camW / 2, camH - 18);
      ctx.restore();
    }

    // Draw bounding boxes mapped back from letterbox → original resolution
    const boxes = displayBoxesRef.current;
    for (const box of boxes) {
      const m = fromLetterbox(box.x, box.y, box.w, box.h, lb);
      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 2;
      ctx.strokeRect(m.x, m.y, m.w, m.h);
      const label = `ID:${box.id} ${box.weight.toFixed(2)}`;
      ctx.font = "600 12px Inter, sans-serif";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(m.x, m.y - 20, tw + 8, 18);
      ctx.fillStyle = "#34d399";
      ctx.fillText(label, m.x + 4, m.y - 6);
    }

    animFrameRef.current = requestAnimationFrame(drawOutput);
  }, [sideLabel]);

  /* ── Send frame to server ── */
  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const sendCanvas = sendCanvasRef.current;
    const ws = wsRef.current;
    const lb = letterboxRef.current;
    if (!video || !sendCanvas || !ws || ws.readyState !== WebSocket.OPEN || !lb) return;

    const ctx = sendCanvas.getContext("2d");
    if (!ctx) return;

    const frameId = frameIdRef.current++;
    if (frameIdRef.current > 0xffffffff) frameIdRef.current = 0;

    // Capture full-res frame into buffer
    createImageBitmap(video).then((bitmap) => {
      if (frameBufferRef.current.size >= FRAME_BUFFER_MAX) {
        const oldestId = Math.min(...frameBufferRef.current.keys());
        frameBufferRef.current.get(oldestId)?.close();
        frameBufferRef.current.delete(oldestId);
      }
      frameBufferRef.current.set(frameId, bitmap);
    });

    // Letterbox-resize to 640×480 send canvas
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, TARGET_W, TARGET_H);
    ctx.drawImage(video, lb.padX, lb.padY, lb.nw, lb.nh);

    // Encode JPEG + prepend 4-byte frameId (uint32 big-endian)
    sendCanvas.toBlob(
      (blob) => {
        if (blob && ws.readyState === WebSocket.OPEN) {
          blob.arrayBuffer().then((jpegBuf) => {
            const header = new ArrayBuffer(4);
            new DataView(header).setUint32(0, frameId, false);
            const combined = new Uint8Array(4 + jpegBuf.byteLength);
            combined.set(new Uint8Array(header), 0);
            combined.set(new Uint8Array(jpegBuf), 4);
            ws.send(combined.buffer);
          });
        }
      },
      "image/jpeg",
      JPEG_QUALITY
    );
  }, []);

  /* ── Start camera + streaming ── */
  const startStreaming = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      const settings = stream.getVideoTracks()[0].getSettings();
      const camW = settings.width || 640;
      const camH = settings.height || 480;
      cameraResRef.current = { w: camW, h: camH };
      setCameraRes(`${camW}×${camH}`);

      outputCanvasRef.current!.width = camW;
      outputCanvasRef.current!.height = camH;
      sendCanvasRef.current!.width = TARGET_W;
      sendCanvasRef.current!.height = TARGET_H;
      letterboxRef.current = computeLetterbox(camW, camH);

      connectWs();
      sendIntervalRef.current = setInterval(sendFrame, 1000 / SEND_FPS);
      animFrameRef.current = requestAnimationFrame(drawOutput);
      setIsStreaming(true);
    } catch (err) {
      console.error("Camera error:", err);
      alert("Could not access camera. Please grant permission.");
    }
  }, [connectWs, sendFrame, drawOutput]);

  /* ── Stop streaming ── */
  const stopStreaming = useCallback(() => {
    if (sendIntervalRef.current) { clearInterval(sendIntervalRef.current); sendIntervalRef.current = null; }
    cancelAnimationFrame(animFrameRef.current);
    const video = videoRef.current;
    if (video) {
      if (video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        video.srcObject = null;
      }
      if (video.src) {
        URL.revokeObjectURL(video.src);
        video.removeAttribute("src");
        video.load();
      }
      video.loop = false;
    }
    wsRef.current?.close();
    wsRef.current = null;

    for (const bmp of frameBufferRef.current.values()) bmp.close();
    frameBufferRef.current.clear();
    displayFrameRef.current?.close();
    displayFrameRef.current = null;
    displayBoxesRef.current = [];
    pendingDrawRef.current = null;
    frameIdRef.current = 0;

    // Reset boundary line and states
    drawnPointsRef.current = [];
    polylineRef.current = null;
    drawStepRef.current = "idle";
    mousePosRef.current = null;
    setDrawStep("idle");
    setPointCount(0);
    setSideLabel(null);

    // Clear output canvas pixels
    const canvas = outputCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }

    setIsStreaming(false);
    setWsConnected(false);
    setDetectionCount(0);
    setServerFps(0);
  }, []);

  /* ── Start video file streaming ── */
  const startVideoFileStreaming = useCallback(async (file: File) => {
    try {
      stopStreaming();

      const video = videoRef.current!;
      video.srcObject = null;
      video.loop = true;
      video.muted = true; // Ensure autoplay is allowed by browser policies
      
      const fileUrl = URL.createObjectURL(file);
      video.src = fileUrl;

      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => {
          resolve();
        };
      });

      await video.play();

      const camW = video.videoWidth || 640;
      const camH = video.videoHeight || 480;
      cameraResRef.current = { w: camW, h: camH };
      setCameraRes(`${camW}×${camH} (File)`);

      outputCanvasRef.current!.width = camW;
      outputCanvasRef.current!.height = camH;
      sendCanvasRef.current!.width = TARGET_W;
      sendCanvasRef.current!.height = TARGET_H;
      letterboxRef.current = computeLetterbox(camW, camH);

      connectWs();
      sendIntervalRef.current = setInterval(sendFrame, 1000 / SEND_FPS);
      animFrameRef.current = requestAnimationFrame(drawOutput);
      setIsStreaming(true);
    } catch (err) {
      console.error("Video file error:", err);
      alert("Could not load or play video file.");
    }
  }, [connectWs, sendFrame, drawOutput, stopStreaming]);

  /* ── Handle Source Switch ── */
  const handleSourceChange = useCallback((source: "camera" | "video") => {
    stopStreaming();
    setInputSource(source);
  }, [stopStreaming]);

  /* ── Handle File Select ── */
  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      startVideoFileStreaming(file);
    }
    // Reset value so selecting the same file again triggers onChange
    e.target.value = "";
  }, [startVideoFileStreaming]);

  /* ── Cleanup on unmount ── */
  useEffect(() => () => stopStreaming(), [stopStreaming]);

  /* ── Restart draw loop when sideLabel changes ── */
  useEffect(() => {
    if (!isStreaming) return;
    cancelAnimationFrame(animFrameRef.current);
    animFrameRef.current = requestAnimationFrame(drawOutput);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [drawOutput, isStreaming]);

  /* ── Finish drawing: extend endpoints and move to pick-side ── */
  const finishDrawing = useCallback(() => {
    const pts = drawnPointsRef.current;
    if (pts.length < 2) return;
    const { w, h } = cameraResRef.current;
    const extended = extendPolylineToBorder(pts, w, h);
    polylineRef.current = extended;
    drawStepRef.current = "pick-side";
    setDrawStep("pick-side");
  }, []);

  /* ── Canvas mouse move ── */
  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const step = drawStepRef.current;
      if (step !== "drawing" && step !== "pick-side") return;
      const canvas = outputCanvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      mousePosRef.current = {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height),
      };
    },
    []
  );

  /* ── Canvas click handler ── */
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const step = drawStepRef.current;
      const canvas = outputCanvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);

      if (step === "drawing") {
        // Double-click detection: if two clicks within 300ms, finish
        const now = Date.now();
        const isDouble = now - lastClickTimeRef.current < 300;
        lastClickTimeRef.current = now;

        if (isDouble && drawnPointsRef.current.length >= 2) {
          // Remove the last point added by the first click of this double-click
          drawnPointsRef.current = drawnPointsRef.current.slice(0, -1);
          finishDrawing();
          return;
        }

        // Enforce max points
        if (drawnPointsRef.current.length >= MAX_POLY_POINTS) return;

        drawnPointsRef.current = [...drawnPointsRef.current, { x, y }];
        setPointCount(drawnPointsRef.current.length);
        return;
      }

      if (step === "pick-side") {
        const poly = polylineRef.current;
        if (!poly || poly.length < 2) return;

        // Convert polyline + click point to 640×480 letterbox space
        // so that parity matches exactly what the backend computes
        const lb = letterboxRef.current;
        if (!lb) return;
        const lbPoints = poly.map((p) => toLetterboxPt(p.x, p.y, lb));
        const lbClick  = toLetterboxPt(x, y, lb);

        // Compute parity in letterbox space — must agree with backend's raycast_parity
        const parity = raycastParity(lbClick.x, lbClick.y, lbPoints);
        const label: "A" | "B" = parity === 0 ? "A" : "B";
        setSideLabel(label);
        drawStepRef.current = "active";
        setDrawStep("active");

        const ws = wsRef.current;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "set_line",
              points: lbPoints.map((p) => [p.x, p.y]),
              side_parity: parity,
            })
          );
        }

        // Resume video file playback once boundary configuration is complete
        if (inputSource === "video") {
          videoRef.current?.play().catch(() => {});
        }
      }
    },
    [finishDrawing, inputSource]
  );

  /* ── Reset line ── */
  const resetLine = useCallback(() => {
    drawnPointsRef.current = [];
    polylineRef.current = null;
    drawStepRef.current = "idle";
    mousePosRef.current = null;
    setDrawStep("idle");
    setPointCount(0);
    setSideLabel(null);
    displayBoxesRef.current = [];

    wsRef.current?.readyState === WebSocket.OPEN &&
      wsRef.current.send(JSON.stringify({ type: "clear_line" }));

    // Resume video file playback if it was paused during drawing
    if (inputSource === "video") {
      videoRef.current?.play().catch(() => {});
    }
  }, [inputSource]);

  /* ── Start drawing mode ── */
  const startDrawing = useCallback(() => {
    drawnPointsRef.current = [];
    polylineRef.current = null;
    mousePosRef.current = null;
    drawStepRef.current = "drawing";
    setDrawStep("drawing");
    setPointCount(0);
    setSideLabel(null);

    // Pause video playback so the user can easily draw on a static frame
    if (inputSource === "video") {
      videoRef.current?.pause();
    }
  }, [inputSource]);

  /* ══════════════════════════════════ */
  /*              Render                */
  /* ══════════════════════════════════ */
  return (
    <div className="app">
      {/* Hidden elements */}
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={sendCanvasRef} style={{ display: "none" }} />

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="header-icon">👁</div>
          <div>
            <h1>Person Detector</h1>
            <div className="header-subtitle">Outdoor CCTV Monitoring</div>
          </div>
        </div>
        <div className="status-group">
          {cameraRes && <span className="status-badge fps-badge">{cameraRes}</span>}
          <span className="status-badge fps-badge">
            ↑ {SEND_FPS}fps &nbsp;↓ {serverFps}fps
          </span>
          <span className={`status-badge ${wsConnected ? "connected" : "disconnected"}`}>
            <span className="status-dot" />
            {wsConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </header>

      {/* Main */}
      <div className="main-content">
        {/* Video Output */}
        <div className="video-area">
          <div
            className={`canvas-wrapper ${
              drawStep === "drawing" || drawStep === "pick-side" ? "drawing-mode" : ""
            }`}
          >
            <canvas
              ref={outputCanvasRef}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onMouseLeave={() => { mousePosRef.current = null; }}
            />
            {!isStreaming && (
              <div className="placeholder-message">
                <div className="placeholder-icon">{inputSource === "camera" ? "📹" : "📁"}</div>
                <p>
                  {inputSource === "camera"
                    ? "Start camera to begin detection"
                    : "Select a video file to begin detection"}
                </p>
              </div>
            )}
            {isStreaming && (
              <div className="canvas-overlay">
                <span className={`overlay-tag ${inputSource === "camera" ? "live" : ""}`}>
                  ● {inputSource === "camera" ? "LIVE" : "FILE"}
                </span>
                {detectionCount > 0 && (
                  <span className="overlay-tag">
                    {detectionCount} person{detectionCount > 1 ? "s" : ""}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Side Panel */}
        <div className="side-panel">
          {/* Input Source Controls */}
          <div className="panel-section">
            <h3>Input Source</h3>
            <div className="source-selector">
              <button
                className={`source-tab ${inputSource === "camera" ? "active" : ""}`}
                onClick={() => handleSourceChange("camera")}
              >
                📹 Webcam
              </button>
              <button
                className={`source-tab ${inputSource === "video" ? "active" : ""}`}
                onClick={() => handleSourceChange("video")}
              >
                📁 Video File
              </button>
            </div>
            <div className="btn-group" style={{ marginTop: "12px" }}>
              {inputSource === "camera" ? (
                !isStreaming ? (
                  <button className="btn btn-primary" onClick={startStreaming} id="btn-start-camera">
                    ▶ Start Camera
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={stopStreaming} id="btn-stop-camera">
                    ■ Stop Camera
                  </button>
                )
              ) : (
                <>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="video/*"
                    style={{ display: "none" }}
                    onChange={handleFileChange}
                  />
                  {!isStreaming ? (
                    <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()} id="btn-upload-video">
                      📁 Select Video File
                    </button>
                  ) : (
                    <button className="btn btn-danger" onClick={stopStreaming} id="btn-stop-video">
                      ■ Stop Video
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Detection Zone */}
          <div className="panel-section">
            <h3>Detection Zone</h3>
            <div className="btn-group">
              {drawStep === "idle" && (
                <button
                  className="btn"
                  onClick={startDrawing}
                  disabled={!isStreaming}
                  id="btn-draw-line"
                >
                  ✏️ Draw Boundary
                </button>
              )}

              {drawStep === "drawing" && (
                <>
                  <div className="line-status">
                    Click to add points. <strong>Double-click</strong> or press
                    Finish to complete.
                    <br />
                    Points: {pointCount} / {MAX_POLY_POINTS}
                  </div>
                  <button
                    className="btn btn-success"
                    onClick={finishDrawing}
                    disabled={pointCount < 2}
                    id="btn-finish-line"
                  >
                    ✓ Finish Line
                  </button>
                </>
              )}

              {drawStep === "pick-side" && (
                <div className="line-status">
                  <strong>Click on the video</strong> to select which side to
                  monitor.
                </div>
              )}

              {drawStep === "active" && (
                <div className="line-status">
                  Boundary <strong>active</strong>. Monitoring{" "}
                  <strong>Side {sideLabel}</strong>.
                </div>
              )}

              {drawStep !== "idle" && (
                <button className="btn btn-danger" onClick={resetLine} id="btn-reset-line">
                  ↺ Reset Boundary
                </button>
              )}
            </div>
          </div>

          {/* Detection Info */}
          <div className="panel-section">
            <h3>Detections</h3>
            <div className="detection-count">
              <span className="label">Persons detected</span>
              <span className="value">{detectionCount}</span>
            </div>
          </div>

          {/* Instructions */}
          <div className="panel-section">
            <h3>How to use</h3>
            <div className="instructions">
              <div className="instruction-step">
                <span className={`step-num ${isStreaming ? "done" : ""}`}>1</span>
                <span>
                  {inputSource === "camera"
                    ? "Start camera to begin streaming"
                    : "Select and load a video file"}
                </span>
              </div>
              <div className="instruction-step">
                <span className={`step-num ${drawStep !== "idle" ? "done" : ""}`}>2</span>
                <span>Draw a multi-point boundary line</span>
              </div>
              <div className="instruction-step">
                <span className={`step-num ${drawStep === "active" ? "done" : ""}`}>3</span>
                <span>Click a side on the video to set detection zone</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
