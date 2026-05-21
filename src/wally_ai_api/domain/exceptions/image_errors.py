class ImageValidationError(Exception):
    def __init__(self, message: str, code: str = "invalid_image") -> None:
        super().__init__(message)
        self.message = message
        self.code = code
