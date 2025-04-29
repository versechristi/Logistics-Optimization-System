# core/distance_calculator.py
# -*- coding: utf-8 -*-
import math

def haversine(coord1, coord2):
    """
    Calculates the great-circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        coord1 (tuple): Tuple of (latitude, longitude) for first point in degrees.
        coord2 (tuple): Tuple of (latitude, longitude) for second point in degrees.

    Returns:
        float: Distance between the two points in kilometers.
               Returns float('inf') if coordinates are invalid.
    """
    if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2:
        print(f"Warning: Invalid coordinates provided to haversine: {coord1}, {coord2}")
        return float('inf') # Return infinity for invalid coordinates

    try:
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

        R = 6371  # Earth radius in kilometers

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance
    except (TypeError, IndexError, ValueError) as e:
        print(f"Error calculating haversine distance between {coord1} and {coord2}: {e}")
        return float('inf')


if __name__ == '__main__':
    # Example usage
    coord_taipei = (25.0330, 121.5654)  # Taipei
    coord_tokyo = (35.6762, 139.6503)   # Tokyo
    distance_tp_tk = haversine(coord_taipei, coord_tokyo)
    print(f"Distance between Taipei and Tokyo: {distance_tp_tk:.2f} km")

    coord_new_york = (40.7128, -74.0060) # New York
    distance_tp_ny = haversine(coord_taipei, coord_new_york)
    print(f"Distance between Taipei and New York: {distance_tp_ny:.2f} km")

    # Test invalid input
    print(f"Distance with invalid input: {haversine(coord_taipei, None)}")
    print(f"Distance with incomplete input: {haversine(coord_taipei, (40.7,))}")