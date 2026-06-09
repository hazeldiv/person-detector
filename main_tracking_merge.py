import cv2, time, argparse, numpy as np, os

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default=None)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(0)
if not args.video:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
TARGET_SIZE = (640, 480)


def resize_letterbox(frame, target):
    h, w = frame.shape[:2]
    scale = min(target[0] / w, target[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((target[1], target[0], 3), dtype=np.uint8)
    x, y = (target[0] - nw) // 2, (target[1] - nh) // 2
    canvas[y:y+nh, x:x+nw] = cv2.resize(frame, (nw, nh))
    return canvas


def centroid(rect):
    return (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)


def point_in_rect(px, py, rect, margin=20):
    x, y, w, h = rect
    return (x - margin <= px <= x + w + margin) and (y - margin <= py <= y + h + margin)


def nms(boxes, scores, iou_thresh=0.45):
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return boxes[np.array(keep)], scores[keep]


def is_human_shaped(contour, min_area=500, max_area=15000):
    area = cv2.contourArea(contour)
    if not (min_area < area < max_area):
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if not (1.2 < h / w < 4.0):
        return False
    hull = cv2.convexHull(contour)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    return 0.4 < solidity < 0.92


class Tracker:
    def __init__(self, max_distance=100, min_hits=3, max_age=10, max_confirmed_age=15, size_change_threshold=0.3, smooth_alpha=0.1, max_size_change=0.2):
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
        xi1, yi1, xi2, yi2 = max(x1, x2), max(y1, y2), min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return inter / (w1 * h1 + w2 * h2 - inter) if (w1 * h1 + w2 * h2 - inter) > 0 else 0

    def _centroid(self, rect):
        return (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)

    def _size_ratio(self, r1, r2):
        a1, a2 = r1[2] * r1[3], r2[2] * r2[3]
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
            rx, ry,
            int(self.smooth_alpha * rw + (1 - self.smooth_alpha) * sw),
            int(self.smooth_alpha * rh + (1 - self.smooth_alpha) * sh)
        )

    def update(self, detections):
        if not detections:
            for t in self.tracks.values():
                t['age'] += 1
            self._cleanup()
            return self._get_confirmed()

        matched, unmatched = set(), []
        for rect, weight in detections:
            det_cx, det_cy = self._centroid(rect)
            best_id, best_dist = None, float('inf')
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                tcx, tcy = self._centroid(track['rect'])
                dist = ((det_cx - tcx) ** 2 + (det_cy - tcy) ** 2) ** 0.5
                if dist < self.max_distance and self._size_ratio(rect, track['rect']) < self.size_change_threshold:
                    if dist < best_dist:
                        best_dist, best_id = dist, tid
            if best_id:
                clamped_rect = self._clamp_size(rect, track.get('smooth_rect', rect))
                smoothed = self._smooth_rect(clamped_rect, track.get('smooth_rect', rect))
                self.tracks[best_id].update({'rect': rect, 'smooth_rect': smoothed, 'weight': weight, 'hits': track['hits'] + 1, 'age': 0, 'centroid': (det_cx, det_cy)})
                matched.add(best_id)
            else:
                unmatched.append((rect, weight))

        for rect, weight in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'rect': rect, 'smooth_rect': rect, 'weight': weight, 'hits': 1, 'age': 0, 'centroid': self._centroid(rect)}

        for tid, track in self.tracks.items():
            if tid not in matched:
                track['age'] += 1

        self._merge_tracks()
        self._cleanup()
        return self._get_confirmed()

    def _merge_tracks(self, iou_thresh=0.3, centroid_thresh=60):
        confirmed = [(tid, t) for tid, t in self.tracks.items() if t['hits'] >= self.min_hits]
        to_remove = set()
        for i, (id1, t1) in enumerate(confirmed):
            if id1 in to_remove:
                continue
            for id2, t2 in confirmed[i + 1:]:
                if id2 in to_remove:
                    continue
                iou = self._iou(t1['rect'], t2['rect'])
                cx1, cy1, cx2, cy2 = t1['centroid'][0], t1['centroid'][1], t2['centroid'][0], t2['centroid'][1]
                cdist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                sratio = self._size_ratio(t1['rect'], t2['rect'])
                if iou > iou_thresh or (cdist < centroid_thresh and sratio < self.size_change_threshold):
                    to_remove.add(id2 if t1['weight'] < t2['weight'] else id1)
        for tid in to_remove:
            del self.tracks[tid]

    def _cleanup(self):
        to_remove = [tid for tid, t in self.tracks.items() if (t['hits'] >= self.min_hits and t['age'] > self.max_confirmed_age) or (t['hits'] < self.min_hits and t['age'] > self.max_age)]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed(self):
        return [(tid, t.get('smooth_rect', t['rect']), t['weight']) for tid, t in self.tracks.items() if t['hits'] >= self.min_hits and t['age'] <= self.max_confirmed_age]


tracker = Tracker(max_distance=80, min_hits=3, max_age=8, max_confirmed_age=10, size_change_threshold=0.35, smooth_alpha=0.9, max_size_change=0.5)

writer = None
if args.video:
    os.makedirs('output/mog2_hog_tracking_merge', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    out_name = os.path.join('output/mog2_hog_tracking_merge', base_name + '.mp4')
    writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'avc1'), 30.0, (640, 480))

fps, frame_count, start_time = 0, 0, time.time()
target_fps, frame_interval, total_frames = 15, 1, 0

if args.video:
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(source_fps / target_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx_global = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if args.video:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx % frame_interval != 0:
            continue
        frame = resize_letterbox(frame, TARGET_SIZE)

    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05, hitThreshold=-1)

    fg_mask = mog2.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_rects = [(cv2.boundingRect(cnt)[0], cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[2], cv2.boundingRect(cnt)[3]) for cnt in contours if cv2.contourArea(cnt) > 2000 and cv2.boundingRect(cnt)[3] > 80 and is_human_shaped(cnt)]
    if args.video:
        cv2.putText(frame, f"{frame_idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    current_detections = []
    nms_rect, nms_weight = nms(rects, weights, 0.2)
    for (x, y, w, h), weight in zip(nms_rect, nms_weight):
        aspect_ratio, area = h / w, w * h
        if 1.5 < aspect_ratio < 2.8 and area > 3000:
            cx, cy = centroid((x, y, w, h))
            if roi_rects and any((x - 10 <= cx <= x + w + 10 and y - 10 <= cy <= y + h + 10) for (x, y, w, h) in roi_rects):
                current_detections.append(((x, y, w, h), weight))

    confirmed_tracks = tracker.update(current_detections)
    for track_id, (x, y, w, h), weight in confirmed_tracks:
        cv2.putText(frame, f"ID:{track_id} {weight:.2f}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if not args.video:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Person Detection + MOG2", frame)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

    if writer:
        writer.write(frame)
        progress = int(frame_idx / total_frames * 40)
        bar = '=' * progress + ' ' * (40 - progress)
        print(f'\r[{bar}] {int(frame_idx / total_frames * 100)}%', end='', flush=True)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

cap.release()
if writer:
    writer.release()
if not args.video:
    cv2.destroyAllWindows()