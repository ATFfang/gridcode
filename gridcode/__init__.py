from .core import GridEncoder, GridBatchConverter

# 单点 API
WGS84_to_gridcode = GridEncoder.WGS84_to_gridcode
gridcode_to_WGS84 = GridEncoder.gridcode_to_WGS84

# 批量 API
WGS84_to_gridcodes = GridBatchConverter.WGS84_to_gridcodes
gridcodes_to_WGS84 = GridBatchConverter.gridcodes_to_WGS84

__all__ = [
    "WGS84_to_gridcode",
    "gridcode_to_WGS84",
    "WGS84_to_gridcodes",
    "gridcodes_to_WGS84",
]
