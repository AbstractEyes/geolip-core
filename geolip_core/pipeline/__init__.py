"""
pipeline — Composed geometric substrates.

Stage interfaces define the contract. Compositions wire behaviors
into reusable blocks. The pipeline is where stages become systems.
"""

from .observer import Input, Mutation, Association, Curation, Distinction, GeoLIP
#from .layer import ConstellationLayer
#from .backbone import GeometricBackbone
