

class ThompsonRoutingStrategy:
    """Thompson Sampling based routing.
    
    Uses Beta distributions to model uncertainty and samples to select models.
    Balances exploration and exploitation naturally.
    """
    
    def __init__(self, models: list[str], seed: int | None = None):
        self.models = models
        self.rng = np.random.RandomState(seed) if seed else np.random
        
        # Beta posteriors: alpha = successes + 1, beta = failures + 1
        self.alpha = {m: 1.0 for m in models}
        self.beta = {m: 1.0 for m in models}
        
    def select_model(self) -> str:
        """Sample from Beta posterior for each model and return best."""
        samples = {m: self.rng.beta(self.alpha[m], self.beta[m]) for m in self.models}
        return max(samples, key=samples.get)
    
    def update(self, model: str, quality: float) -> None:
        """Update posterior with observed quality."""
        if quality > 0:
            self.alpha[model] += 1
        else:
            self.beta[model] += 1
    
    def get_posteriors(self) -> dict[str, tuple[float, float]]:
        """Return current (alpha, beta) for each model."""
        return {m: (self.alpha[m], self.beta[m]) for m in self.models}
