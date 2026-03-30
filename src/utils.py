import cv2
import numpy as np

def ccw(A, B, C):
    """Counter-Clockwise helper for intersection."""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    """
    Returns True if line segment AB intersects line segment CD.
    A, B: Vehicle Movement Vector (Prev, Curr)
    C, D: Trap Line (Point 1, Point 2)
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_line_intersection(p0, p1, p2, p3):
    """
    Returns the exact (x, y) intersection point and the ratio 't'.
    p0-p1: Vehicle Vector (Prev -> Curr)
    p2-p3: Trap Line (Line Start -> Line End)
    
    Returns:
        (x, y): The pixel coordinate of the intersection
        t: The fraction (0.0 to 1.0) of the vector length where intersection occurred.
           Used for sub-frame timing (e.g. t=0.5 means it happened exactly halfway between frames).
    """
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    denom = (-s2_x * s1_y + s1_x * s2_y)
    
    # Avoid division by zero (Parallel lines)
    if denom == 0: 
        return None, None 

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denom
    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denom

    if (s >= 0 and s <= 1 and t >= 0 and t <= 1):
        # Collision detected
        i_x = p0[0] + (t * s1_x)
        i_y = p0[1] + (t * s1_y)
        return (i_x, i_y), t
        
    return None, None

def is_point_in_poly(point, poly_points):
    """Checks if point is inside the ROI polygon."""
    if len(poly_points) < 3: return False
    poly = np.array(poly_points, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0

def get_box_center(box):
    """Bottom-center of bounding box."""
    x1, y1, x2, y2 = map(int, box)
    return (x1 + x2) // 2, y2