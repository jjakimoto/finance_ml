from scipy.stats import rv_continuous


class LogUniformGen(rv_continuous):
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a=1, b=np.exp(1)):
    return LogUniformGen(a=a, b=b, name='log_uniform')