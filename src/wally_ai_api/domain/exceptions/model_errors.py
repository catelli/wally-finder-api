class ModelNotLoadedError(Exception):
    def __init__(self, message: str = "Model is not loaded") -> None:
        super().__init__(message)
        self.message = message
        self.code = "model_not_loaded"
