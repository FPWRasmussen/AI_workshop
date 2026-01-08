from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_turbines import WindTurbine

class SimpleDTU10MW(WindTurbine):
    def __init__(self, method='linear'):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(self, name='wt', diameter=1, hub_height=1,
            powerCtFunction=PowerCtTabular(
                ws=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                power=[0.0, 204.5, 663.3, 1333.4, 2305.6, 3469.6, 4993.1, 6885.7, 9164.9, 10139.1, 10148.0, 10139.3, 10181.6, 10141.8, 10139.9, 10139.8, 10152.1, 10145.8, 10143.7, 10141.1, 10139.4, 10143.4, 10135.8],
                power_unit="kw",
                ct=[0.0, 0.923, 0.919, 0.904, 0.858, 0.814, 0.814, 0.814, 0.814, 0.577, 0.419, 0.323, 0.259, 0.211, 0.175, 0.148, 0.126, 0.109, 0.095, 0.084, 0.074, 0.066, 0.059],
                method=method))
