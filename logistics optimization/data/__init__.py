# data/__init__.py
#   -*- coding: utf-8 -*-

from .data_generator import generate_locations, generate_demand
#   If you implement Solomon loading, uncomment the line below
#   from .solomon_parser import load_solomon_data

__all__ = [
    "generate_locations",
    "generate_demand",
    #   "load_solomon_data",  #   Add if implemented
]