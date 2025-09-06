import cupy as cp
import numpy as np

class VegetationIndices:
    """
    This class calculates various vegetation indices.
    """

    def __init__(self):
        pass

    def normalize_index(self, index):
        """
        Normalizes a vegetation index between -1 and 1.

        Args:
            index: A CuPy array containing the vegetation index values.

        Returns:
            A CuPy array containing the normalized values.
        """
        min_value = cp.min(index)
        max_value = cp.max(index)
        normalized_index = 2 * (index - min_value) / (max_value - min_value) - 1
        return normalized_index

    def fill_sus_values(self, index):
        """
        Replaces infinity and NaN values in the index with 0.

        Args:
            index: A CuPy array containing the vegetation index values.

        Returns:
            A CuPy array with suspicious values replaced by 0.
        """
        # Use cupy's isinf and isnan functions
        suspicious_mask = cp.logical_or(cp.isinf(index), cp.isnan(index))
        index[suspicious_mask] = 0
        return index

    def calculate_msavi(self, red, green, blue, red_edge, nir):
        """
        Calculates the MSAVI index.

        Args:
            red: A CuPy array containing the red band values.
            nir: A CuPy array containing the near-infrared band values.

        Returns:
            A CuPy array containing the MSAVI values.
        """
        msavi = (2 * nir + 1 - cp.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2
        msavi = self.fill_sus_values(msavi)
        return self.normalize_index(msavi)

    def calculate_ndvi(self, red, green, blue, red_edge, nir):
        """
        Calculates the NDVI index.

        Args:
            red: A CuPy array containing the red band values.
            nir: A CuPy array containing the near-infrared band values.

        Returns:
            A CuPy array containing the NDVI values.
        """
        ndvi = (nir - red) / (nir + red)
        ndvi = self.fill_sus_values(ndvi)
        return self.normalize_index(ndvi)
    
    def calculate_osavi_1(self, red, green, blue, red_edge, nir):
        osavi = (nir - red) / (nir + red + 0.16)
        osavi = self.fill_sus_values(osavi)
        return self.normalize_index(osavi)
    
    # Rondeaux et al., 1996
    def calculate_osavi_2(self, red, green, blue, red_edge, nir):
        osavi = (1 + 1.16) * (nir - red) / (nir + red + 0.16)
        osavi = self.fill_sus_values(osavi)
        return self.normalize_index(osavi)
    
    def calculate_ndre(self, red, green, blue, red_edge, nir):
        ndre = (nir - red_edge) / (nir + red_edge)
        ndre = self.fill_sus_values(ndre)
        return self.normalize_index(ndre)

    def calculate_exg(self, red, green, blue, red_edge, nir):
        exg = (2*green - red - blue) / (2*green + red + blue)
        exg = self.fill_sus_values(exg)
        return self.normalize_index(exg)

    def calculate_evi(self, red, green, blue, red_edge, nir, g = 2.5, c1 = 6, c2 = 7.5, l = 1):
        evi = g * ((nir - red) / (nir + (c1 * red) - (c2 * blue) + l))
        evi = self.fill_sus_values(evi)   
        return self.normalize_index(evi)

    def calculate_nidi(self, red, green, blue, red_edge, nir):
        nidi = (nir - red_edge) / (nir + red_edge)
        nidi = self.fill_sus_values(nidi)
        return self.normalize_index(nidi)

    def calculate_gndvi(self, red, green, blue, red_edge, nir):
        gndvi = (nir - green) / (nir + green)
        gndvi = self.fill_sus_values(gndvi)
        return self.normalize_index(gndvi)

    def calculate_triangular_vi_1(self, red, green, blue, red_edge, nir):
        """
        Calculates the Triangular Vegetation Index (TVI).

        Args:
            red, green, blue, red_edge, nir: CuPy arrays containing the corresponding band values.

        Returns:
            A CuPy array containing the TVI values.
        """
        tvi = 0.5 * (120 * (red_edge - green) - 200 * (red - green))
        tvi = self.fill_sus_values(tvi)
        return self.normalize_index(tvi)
    
    # Brodge & Leblanc, 2001
    def calculate_triangular_vi_2(self, red, green, blue, red_edge, nir):
        tvi = 0.5 * (120 * (nir - green) - 200 * (red - green))
        tvi = self.fill_sus_values(tvi)
        return self.normalize_index(tvi)
    
    # Haboudance et al., 2004
    def calculate_modified_triangular_vi(self, red, green, blue, red_edge, nir):
        modified_tvi = 1.5 * (1.2 * (nir - green) - 2.5 * (red - green)) / cp.sqrt((2 * nir + 1)^2 - (6 * nir - 5 * cp.sqrt(red)) - 0.5)
        modified_tvi = self.fill_sus_values(modified_tvi)
        return self.normalize_index(modified_tvi)

    # Green Soil Adjusted Vegetation Index: Sripada et al., 2006
    def calculate_gsavi(self, red, green, blue, red_edge, nir):
        gsavi = (nir - green) / (nir + green + 0.5) * (1 + 0.5)
        gsavi = self.fill_sus_values(gsavi)
        return self.normalize_index(gsavi)

    # Red Edge Inflection Point
    def calculate_reip(self, red, green, blue, red_edge, nir):
        reip = (0.5 * (red + nir) - red_edge) / (nir - red_edge)
        reip = self.fill_sus_values(reip)
        return self.normalize_index(reip)

    def get_vi_function(self, name):
        """
        Retrieves the function corresponding to a specific vegetation index name.

        Args:
            name: The name of the vegetation index (e.g., 'NDVI', 'SAVI').

        Returns:
            The function object for the requested vegetation index, or None if not found.

        Raises:
            ValueError: If the provided name is not a valid vegetation index.
        """

        available_vi = {
            # Greenery
            'NDVI': self.calculate_ndvi,
            'NDRE': self.calculate_ndre,
            'EXG': self.calculate_exg,
            'EVI': self.calculate_evi,
            'NIDI': self.calculate_nidi,
            'TVI1': self.calculate_triangular_vi_1,
            'TVI2': self.calculate_triangular_vi_2,
            'MODIFIED_TVI': self.calculate_modified_triangular_vi,
            'RedEdge_InflectedPoint': self.calculate_reip,
            # Soil
            'MSAVI': self.calculate_msavi,
            'OSAVI1': self.calculate_osavi_1,
            'OSAVI2': self.calculate_osavi_2,
            'GNDVI': self.calculate_gndvi, # Green Normalized Difference Vegetation
            'GSAVI': self.calculate_gsavi,
        }

        if name not in available_vi:
            raise ValueError(f"Invalid vegetation index name: {name}")

        return available_vi[name]