import math
import numpy as np
import os
from multiprocessing import Pool
from typing import List, Tuple


class GridEncoder:
    """
    Core module for geographic grid encoding/decoding (GB/T 16831 standard).
    Provides conversion between WGS84 coordinates and grid codes.
    """

    def __init__(self) -> None:
        # Grid span list, from degrees to sub-arcsecond resolution
        self._grid_span_list: List[float] = [
            256, 128, 64, 32, 16, 8, 4, 2, 1,
            32 / 60, 16 / 60, 8 / 60, 4 / 60, 2 / 60, 1 / 60,
            32 / 3600, 16 / 3600, 8 / 3600, 4 / 3600, 2 / 3600, 1 / 3600,
            1 / 2 / 3600, 1 / 4 / 3600, 1 / 8 / 3600, 1 / 16 / 3600,
            1 / 32 / 3600, 1 / 64 / 3600, 1 / 128 / 3600,
            1 / 256 / 3600, 1 / 512 / 3600, 1 / 1024 / 3600, 1 / 2048 / 3600
        ]

    # -------------------- Encode API --------------------

    @staticmethod
    def _decimal_to_dms(decimal: float) -> Tuple[int, int, int, float]:
        """Decimal degrees → DMS tuple."""
        degrees = int(decimal)
        minutes = int((abs(decimal) - abs(degrees)) * 60)
        seconds = (abs(decimal) - abs(degrees)) * 60
        seconds = (seconds - minutes) * 60
        seconds_fractional, seconds_integer = math.modf(seconds)
        return degrees, minutes, int(seconds_integer), seconds_fractional

    @staticmethod
    def _dms_to_bits(dms: Tuple[int, int, int, float]) -> str:
        """DMS tuple → 31-bit binary string."""
        degrees, minutes, seconds_integer, seconds_fractional = dms
        degrees_bin = f"{degrees:08b}"
        minutes_bin = f"{minutes:06b}"
        seconds_integer_bin = f"{seconds_integer:06b}"
        seconds_fractional_scaled = int(seconds_fractional * 2048)
        seconds_fractional_bin = f"{seconds_fractional_scaled:011b}"

        return ''.join([degrees_bin, minutes_bin, seconds_integer_bin, seconds_fractional_bin])

    def _morton_cross(self, latitude: float, longitude: float) -> str:
        """Interleave latitude/longitude bits (Morton encoding)."""
        lat_bits = self._dms_to_bits(self._decimal_to_dms(latitude))
        lon_bits = self._dms_to_bits(self._decimal_to_dms(longitude))

        assert len(lat_bits) == 31 and len(lon_bits) == 31, \
            "Both binary strings must be 31 bits long."

        interleaved_bits = []
        for i in range(31):
            interleaved_bits.append(lat_bits[i])
            interleaved_bits.append(lon_bits[i])

        bit2 = ''.join(interleaved_bits)
        bit4 = ''.join(str(int(bit2[i:i + 2], 2)) for i in range(0, len(bit2), 2))
        return bit4

    def _encode_latlon(self, latitude: float, longitude: float, m: int) -> str:
        """Latitude/Longitude → Grid code (G0 prefix)."""
        bit4 = self._morton_cross(latitude, longitude)
        grid_code = bit4[:m - 32]
        return f"G0{grid_code}"

    def _encode_elevation(self, elevation: float, m: int) -> str:
        """Elevation → Grid code (H prefix)."""
        theta_0 = math.pi / 180
        theta = self._grid_span_list[m - 1] * math.pi / 180
        r_0 = 6378137  # Earth's mean radius (WGS84)
        H = elevation

        n = (theta_0 / theta) * math.log((H + r_0) / r_0, (1 + theta_0))
        n = max(0, math.floor(n))
        n_bin = bin(n)[2:]

        if len(n_bin) < m:
            n_bin = '0' * (m - len(n_bin)) + n_bin

        return f"H{n_bin}"

    # -------------------- Decode API --------------------

    @staticmethod
    def _decode_dms_bits(binary_str: str) -> float:
        """Binary string → Decimal degrees."""
        degrees_bin = binary_str[0:8]
        minutes_bin = binary_str[8:14]
        seconds_integer_bin = binary_str[14:20]
        seconds_fractional_bin = binary_str[20:31]

        degrees = int(degrees_bin, 2)
        minutes = int(minutes_bin, 2)
        seconds_integer = int(seconds_integer_bin, 2)
        seconds_fractional_scaled = int(seconds_fractional_bin, 2)

        seconds_fractional = seconds_fractional_scaled / 2048
        total_seconds = seconds_integer + seconds_fractional
        total_minutes = minutes + total_seconds / 60
        decimal = abs(degrees) + total_minutes / 60
        return decimal

    def _decode_latlon(self, gridcode: str) -> Tuple[float, float]:
        """Grid code → (lat, lon)."""
        gridcode_number = gridcode[2:]
        gridcode_padded = gridcode_number.ljust(31, '0')

        bit2 = ''.join(bin(int(c, 4))[2:].zfill(2) for c in gridcode_padded)

        lat_bits = [bit2[i * 2] for i in range(31)]
        lon_bits = [bit2[i * 2 + 1] for i in range(31)]

        lat_code = ''.join(lat_bits)
        lon_code = ''.join(lon_bits)

        lat = self._decode_dms_bits(lat_code)
        lon = self._decode_dms_bits(lon_code)
        return lat, lon

    def _decode_elevation(self, elevationcode: str, m: int) -> float:
        """Elevation code → Elevation (meters)."""
        n_binary = elevationcode[1:]  # 去掉 'H'
        n = int(n_binary, 2)

        theta_0 = math.pi / 180
        theta = self._grid_span_list[m - 1] * math.pi / 180
        r_0 = 6378137

        H = r_0 * (math.exp((n * theta) / theta_0 * math.log(1 + theta_0)) - 1)
        return H

    # -------------------- Public API --------------------

    @staticmethod
    def WGS84_to_gridcode(longitude: float, latitude: float, elevation: float, m: int) -> List[str]:
        """
        Convert WGS84 (lon, lat, elevation) → Grid codes.
        Returns [grid_code, elevation_code].
        """
        encoder = GridEncoder()
        grid_code = encoder._encode_latlon(latitude, longitude, m)
        elevation_code = encoder._encode_elevation(elevation, m)
        return [grid_code, elevation_code]

    @staticmethod
    def gridcode_to_WGS84(gridcode: str, elevationcode: str, m: int) -> List[float]:
        """
        Convert Grid codes → WGS84 (lon, lat, elevation).
        Returns [lon, lat, elevation].
        """
        encoder = GridEncoder()
        lat, lon = encoder._decode_latlon(gridcode)
        elevation = encoder._decode_elevation(elevationcode, m)
        return [lon, lat, elevation]

class GridBatchConverter:
    """
    Batch conversion utilities for GridEncoder.
    Supports single-thread and multiprocessing modes.
    """

    @staticmethod
    def _wgs2grid_worker(arr_chunk: np.ndarray, m: int) -> List[List[str]]:
        """Worker: WGS84 → GridCode for a chunk of data."""
        results = []
        for row in arr_chunk:
            idx = int(row[0])
            lon, lat, elev = row[1:]
            results.append([idx] + GridEncoder.WGS84_to_gridcode(lon, lat, elev, m))
        return results

    @staticmethod
    def _grid2wgs_worker(arr_chunk: np.ndarray, m: int) -> List[List[float]]:
        """Worker: GridCode → WGS84 for a chunk of data."""
        results = []
        for row in arr_chunk:
            idx = int(row[0])
            gridcode, elev_code = row[1:]
            results.append([idx] + GridEncoder.gridcode_to_WGS84(gridcode, elev_code, m))
        return results

    # -------------------- Public API --------------------

    @staticmethod
    def WGS84_to_gridcodes(arr: np.ndarray, m: int, multiprocessing_enabled: bool = True, n_jobs: int = None) -> np.ndarray:
        """
        Batch conversion: WGS84 → Grid codes.
        Input: arr = Nx3 numpy array [[lon, lat, elevation], ...]
        Output: Nx2 numpy array [[gridcode, elev_code], ...]
        """
        index = np.arange(1, arr.shape[0] + 1).reshape(-1, 1)
        arr_indexed = np.hstack((index, arr))

        if multiprocessing_enabled:
            chunk_num = n_jobs or min(arr.shape[0], os.cpu_count() or 4)
            split_arrays = np.array_split(arr_indexed, chunk_num, axis=0)
            args = [(chunk, m) for chunk in split_arrays]

            with Pool(chunk_num) as pool:
                results = pool.starmap(GridBatchConverter._wgs2grid_worker, args)

            merged = [item for sublist in results for item in sublist]
        else:
            merged = GridBatchConverter._wgs2grid_worker(arr_indexed, m)

        merged = np.array(merged, dtype=object)
        merged = merged[np.argsort(merged[:, 0].astype(int))]
        return merged[:, 1:]

    @staticmethod
    def gridcodes_to_WGS84(arr: np.ndarray, m: int, multiprocessing_enabled: bool = True, n_jobs: int = None) -> np.ndarray:
        """
        Batch conversion: Grid codes → WGS84.
        Input: arr = Nx2 numpy array [[gridcode, elev_code], ...]
        Output: Nx3 numpy array [[lon, lat, elevation], ...]
        """
        index = np.arange(1, arr.shape[0] + 1).reshape(-1, 1)
        arr_indexed = np.hstack((index, arr))

        if multiprocessing_enabled:
            chunk_num = n_jobs or min(arr.shape[0], os.cpu_count() or 4)
            split_arrays = np.array_split(arr_indexed, chunk_num, axis=0)
            args = [(chunk, m) for chunk in split_arrays]

            with Pool(chunk_num) as pool:
                results = pool.starmap(GridBatchConverter._grid2wgs_worker, args)

            merged = [item for sublist in results for item in sublist]
        else:
            merged = GridBatchConverter._grid2wgs_worker(arr_indexed, m)

        merged = np.array(merged, dtype=object)
        merged = merged[np.argsort(merged[:, 0].astype(int))]
        return merged[:, 1:]
