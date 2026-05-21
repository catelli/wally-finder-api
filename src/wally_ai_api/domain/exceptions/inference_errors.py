class InferenceError(Exception):
    def __init__(self, message: str, code: str = "inference_failed") -> None:
        super().__init__(message)
        self.message = message
        self.code = code
