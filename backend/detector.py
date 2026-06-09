"""
Detection logic extracted from main_tracking_merge.py.
HOG + MOG2 person detection with tracking.
"""

import cv2
import numpy as np


def nms(boxes, scores, iou_thresh=0.45):
    """Non-Maximum Suppression."""
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return boxes[np.array(keep)], scores[keep]


def is_human_shaped(contour, min_area=500, max_area=15000):
    """Check if a contour has human-like proportions."""
    area = cv2.contourArea(contour)
    if not (min_area < area < max_area):
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if not (1.2 < h / w < 4.0):
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    return 0.4 < solidity < 0.92


def centroid(rect):
    """Get centroid of a bounding rect (x, y, w, h)."""
    return (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)


def raycast_parity(px: float, py: float, points: list) -> int:
    """
    Horizontal ray-cast parity test.
    Fires a ray rightward from (px, py) and counts polyline segment crossings.
    Returns 0 or 1. Matches the JS frontend implementation.
    """
    count = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        if (y1 <= py < y2) or (y2 <= py < y1):
            t = (py - y1) / (y2 - y1)
            ix = x1 + t * (x2 - x1)
            if ix > px:
                count += 1
    return count % 2


class Tracker:
    """Multi-object tracker using centroid distance and IoU matching."""

    def __init__(
        self,
        max_distance=80,
        min_hits=3,
        max_age=8,
        max_confirmed_age=10,
        size_change_threshold=0.35,
        smooth_alpha=0.9,
        max_size_change=0.5,
    ):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_age = max_age
        self.max_confirmed_age = max_confirmed_age
        self.size_change_threshold = size_change_threshold
        self.smooth_alpha = smooth_alpha
        self.max_size_change = max_size_change

    def _iou(self, r1, r2):
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    def _centroid(self, rect):
        return (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)

    def _size_ratio(self, r1, r2):
        a1 = r1[2] * r1[3]
        a2 = r2[2] * r2[3]
        return abs(a1 - a2) / max(a1, a2)

    def _clamp_size(self, raw_rect, prev_rect):
        rx, ry, rw, rh = raw_rect
        _, _, pw, ph = prev_rect
        max_w = pw * (1 + self.max_size_change)
        min_w = pw * (1 - self.max_size_change)
        max_h = ph * (1 + self.max_size_change)
        min_h = ph * (1 - self.max_size_change)
        cw = max(min_w, min(max_w, rw))
        ch = max(min_h, min(max_h, rh))
        return (rx, ry, int(cw), int(ch))

    def _smooth_rect(self, raw_rect, smoothed_rect):
        rx, ry, rw, rh = raw_rect
        _, _, sw, sh = smoothed_rect
        return (
            rx,
            ry,
            int(self.smooth_alpha * rw + (1 - self.smooth_alpha) * sw),
            int(self.smooth_alpha * rh + (1 - self.smooth_alpha) * sh),
        )

    def update(self, detections):
        if not detections:
            for t in self.tracks.values():
                t["age"] += 1
            self._cleanup()
            return self._get_confirmed()

        matched = set()
        unmatched = []

        for rect, weight in detections:
            det_cx, det_cy = self._centroid(rect)
            best_id = None
            best_dist = float("inf")
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                tcx, tcy = self._centroid(track["rect"])
                dist = ((det_cx - tcx) ** 2 + (det_cy - tcy) ** 2) ** 0.5
                if (
                    dist < self.max_distance
                    and self._size_ratio(rect, track["rect"])
                    < self.size_change_threshold
                ):
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid

            if best_id:
                track = self.tracks[best_id]
                clamped_rect = self._clamp_size(
                    rect, track.get("smooth_rect", rect)
                )
                smoothed = self._smooth_rect(
                    clamped_rect, track.get("smooth_rect", rect)
                )
                self.tracks[best_id].update(
                    {
                        "rect": rect,
                        "smooth_rect": smoothed,
                        "weight": weight,
                        "hits": track["hits"] + 1,
                        "age": 0,
                        "centroid": (det_cx, det_cy),
                    }
                )
                matched.add(best_id)
            else:
                unmatched.append((rect, weight))

        for rect, weight in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "rect": rect,
                "smooth_rect": rect,
                "weight": weight,
                "hits": 1,
                "age": 0,
                "centroid": self._centroid(rect),
            }

        for tid, track in self.tracks.items():
            if tid not in matched:
                track["age"] += 1

        self._merge_tracks()
        self._cleanup()
        return self._get_confirmed()

    def _merge_tracks(self, iou_thresh=0.3, centroid_thresh=60):
        confirmed = [
            (tid, t) for tid, t in self.tracks.items() if t["hits"] >= self.min_hits
        ]
        to_remove = set()
        for i, (id1, t1) in enumerate(confirmed):
            if id1 in to_remove:
                continue
            for id2, t2 in confirmed[i + 1 :]:
                if id2 in to_remove:
                    continue
                iou = self._iou(t1["rect"], t2["rect"])
                cx1, cy1 = t1["centroid"]
                cx2, cy2 = t2["centroid"]
                cdist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                sratio = self._size_ratio(t1["rect"], t2["rect"])
                if iou > iou_thresh or (
                    cdist < centroid_thresh and sratio < self.size_change_threshold
                ):
                    to_remove.add(id2 if t1["weight"] < t2["weight"] else id1)
        for tid in to_remove:
            del self.tracks[tid]

    def _cleanup(self):
        to_remove = [
            tid
            for tid, t in self.tracks.items()
            if (t["hits"] >= self.min_hits and t["age"] > self.max_confirmed_age)
            or (t["hits"] < self.min_hits and t["age"] > self.max_age)
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed(self):
        return [
            (tid, t.get("smooth_rect", t["rect"]), t["weight"])
            for tid, t in self.tracks.items()
            if t["hits"] >= self.min_hits and t["age"] <= self.max_confirmed_age
        ]


class PersonDetector:
    """Combines HOG + MOG2 detection pipeline, matching the prototype logic."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self.tracker = Tracker(
            max_distance=80,
            min_hits=3,
            max_age=8,
            max_confirmed_age=10,
            size_change_threshold=0.35,
            smooth_alpha=0.9,
            max_size_change=0.5,
        )
        # Detection zone config: None or dict with 'points' (list of [x,y]) and 'side_parity' (0 or 1)
        self.detection_zone = None

    def set_detection_line(self, points: list, side_parity: int):
        """
        Set the detection boundary polyline.
        points: list of [x, y] in 640×480 letterbox space.
        side_parity: 0 or 1 — the raycast parity of the user's chosen side.
        """
        self.detection_zone = {"points": points, "side_parity": side_parity}

    def clear_detection_line(self):
        self.detection_zone = None

    def _is_on_selected_side(self, cx, cy):
        """Check if a centroid is on the selected side of the detection polyline."""
        if self.detection_zone is None:
            return True
        parity = raycast_parity(cx, cy, self.detection_zone["points"])
        return parity == self.detection_zone["side_parity"]

    def detect(self, frame):
        """
        Run detection on a 640x480 frame.
        Returns list of dicts: [{"id": int, "x": int, "y": int, "w": int, "h": int, "weight": float}]
        """
        # HOG detection
        rects, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05, hitThreshold=-1
        )

        # MOG2 foreground mask
        fg_mask = self.mog2.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Contour-based ROI filtering
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        roi_rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 2000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bh > 80 and is_human_shaped(cnt):
                    roi_rects.append((bx, by, bw, bh))

        # Merge HOG detections with MOG2 ROIs
        current_detections = []
        if len(rects) > 0:
            nms_rect, nms_weight = nms(rects, weights, 0.2)
            for (x, y, w, h), weight in zip(nms_rect, nms_weight):
                aspect_ratio = h / w
                area = w * h
                if 1.5 < aspect_ratio < 2.8 and area > 3000:
                    cx, cy = centroid((x, y, w, h))
                    # Check if centroid falls within any ROI rect
                    if roi_rects and any(
                        (rx - 10 <= cx <= rx + rw + 10 and ry - 10 <= cy <= ry + rh + 10)
                        for (rx, ry, rw, rh) in roi_rects
                    ):
                        current_detections.append(((x, y, w, h), float(weight)))

        # Update tracker
        confirmed_tracks = self.tracker.update(current_detections)

        # Filter by detection line
        results = []
        for track_id, (x, y, w, h), weight in confirmed_tracks:
            cx, cy = centroid((x, y, w, h))
            if self._is_on_selected_side(cx, cy):
                results.append(
                    {
                        "id": int(track_id),
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "weight": round(float(weight), 2),
                    }
                )

        return results
