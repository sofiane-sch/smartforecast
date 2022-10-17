from .utils import *
from scipy.signal import periodogram


class Timeseries:
    def __init__(
        self,
        start: float = 0,
        end: float = 10,
        dt: float = 1,
        seasonalities: list[dict] = [
            {"frequency": 1, "mode": "multiplicative", "amplitude": 1, "phi": 0}
        ],
        trend: dict = {
            "type": "exponential",
            "coefficient": 1,
            "mode": "multiplicative",
        },
        level: float = 0,
        add_noise: bool = False,
        gaussian_noise: dict = {"mu": 0, "sigma": 1},
    ) -> None:
        self.start = start
        self.end = end
        self.dt = dt
        self.seasonalities = seasonalities
        self.trend = trend
        self.add_noise = add_noise
        self.gaussian_noise = gaussian_noise
        self.data = None
        self.time = self.generate_time()
        self.level = level

    @property
    def ts(self):
        x = 0
        x = self.generate_trend()
        seasonalities = []
        for season in self.seasonalities:
            if season["mode"] == "multiplicative":
                x *= self.generate_seasonality(
                    coeff=season["amplitude"],
                    frequency=season["frequency"],
                    phi=season["phi"],
                )
            elif season["mode"] == "additive":
                x += self.generate_seasonality(
                    coeff=season["amplitude"],
                    frequency=season["frequency"],
                    phi=season["phi"],
                )

        if self.add_noise:
            x += self.generate_white_noise(
                mu=self.gaussian_noise["mu"],
                sigma=self.gaussian_noise["sigma"],
                n_sample=len(self.time),
            )
        x += self.level
        return x

    @property
    def autocorrelation(self):
        return autocorrelation_function(self.ts)

    def periodogram(self):
        return periodogram(self.ts, 1 / self.dt)

    def generate_time(self):
        x = np.arange(self.start, self.end + self.dt, self.dt)
        return x

    def generate_trend(self):
        trend_method = {
            "exponential": exponential_trend,
            "linear": linear_trend,
        }
        return trend_method[self.trend["type"]](
            x=self.time,
            coeff=self.trend["coefficient"],
        )

    def moving_average(self, w):
        return np.convolve(self.ts, np.ones(w), "valid") / w

    def generate_white_noise(self, mu: float = 0, sigma: float = 1, n_sample: int = 10):
        self._noise = np.random.normal(mu, sigma, n_sample)
        return self._noise

    def generate_seasonality(self, frequency, phi, coeff):
        return seasonality(x=self.time, freq=frequency, phi=phi, coeff=coeff)
