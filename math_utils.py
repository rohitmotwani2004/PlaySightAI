import cv2
import numpy as np

class CourtTransformer:
    def __init__(self):
        self.court_width = 6.1
        self.court_length = 13.4
        self.scale = 35
        self.map_w = int(self.court_width * self.scale)
        self.map_h = int(self.court_length * self.scale)
        self.margin = 50 

        self.dst_points = np.float32([
            [self.margin, self.margin],
            [self.margin + self.map_w, self.margin],
            [self.margin + self.map_w, self.margin + self.map_h],
            [self.margin, self.margin + self.map_h]
        ])
        self.matrix = None
        self.prev_corners = None

    def get_quadrant_corners(self, pts):
        if pts is None or len(pts) < 4: return None
        center_x, center_y = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        tl_c = pts[(pts[:, 0] < center_x) & (pts[:, 1] < center_y)]
        tr_c = pts[(pts[:, 0] > center_x) & (pts[:, 1] < center_y)]
        bl_c = pts[(pts[:, 0] < center_x) & (pts[:, 1] > center_y)]
        br_c = pts[(pts[:, 0] > center_x) & (pts[:, 1] > center_y)]

        if any(len(c) == 0 for c in [tl_c, tr_c, bl_c, br_c]): return self.prev_corners

        tl = tl_c[np.argmin(np.sum(tl_c, axis=1))]
        tr = tr_c[np.argmin(np.diff(tr_c, axis=1))]
        br = br_c[np.argmax(np.sum(br_c, axis=1))]
        bl = bl_c[np.argmax(np.diff(bl_c, axis=1))]

        current_corners = np.array([tl, tr, br, bl], dtype=np.float32)
        if self.prev_corners is not None:
            current_corners = self.prev_corners * 0.9 + current_corners * 0.1
        self.prev_corners = current_corners
        return current_corners

    def calculate_matrix(self, src_points):
        best_corners = self.get_quadrant_corners(src_points)
        if best_corners is None: return False
        self.matrix = cv2.getPerspectiveTransform(best_corners, self.dst_points)
        return True

    def transform_point(self, x, y):
        if self.matrix is None: return None
        point_array = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_array, self.matrix)
        return transformed_point[0][0]

    def draw_minimap(self, transformed_data):
        h, w = self.map_h + 2 * self.margin, self.map_w + 2 * self.margin
        minimap = np.full((h, w, 3), (34, 85, 34), dtype=np.uint8)
        white = (255, 255, 255)
        
        cv2.rectangle(minimap, (self.margin, self.margin), (self.margin + self.map_w, self.margin + self.map_h), white, 2)
        mid_y = self.margin + (self.map_h // 2)
        cv2.line(minimap, (self.margin, mid_y), (self.margin + self.map_w, mid_y), (0, 255, 255), 2)
        
        ssl = int(1.98 * self.scale)
        cv2.line(minimap, (self.margin, mid_y - ssl), (self.margin + self.map_w, mid_y - ssl), white, 1)
        cv2.line(minimap, (self.margin, mid_y + ssl), (self.margin + self.map_w, mid_y + ssl), white, 1)
        
        cx = self.margin + (self.map_w // 2)
        cv2.line(minimap, (cx, self.margin), (cx, mid_y - ssl), white, 1)
        cv2.line(minimap, (cx, mid_y + ssl), (cx, self.margin + self.map_h), white, 1)

        for p in transformed_data:
            # FIXED: accessing p['pos'] instead of p[0]
            px, py = int(p['pos'][0]), int(p['pos'][1])
            if 0 <= px < w and 0 <= py < h:
                color = (0, 255, 255) if p['shot'] == "SMASH!" else (0, 255, 0)
                if p['shot'] == "LIFT": color = (255, 0, 0) # Lifts are Blue on map
                cv2.circle(minimap, (px, py), 10, color, -1)
                cv2.putText(minimap, str(p['id']), (px-5, py+5), 1, 0.8, white, 1)
        return minimap