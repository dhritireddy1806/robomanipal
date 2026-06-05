class Validator:
    def validate(self,data):
        if "summary" not in data:
            raise ValueError(
                "Missing summary field"
            )
        if "keywords" not in data:
            raise ValueError(
                "Missing keywords field"
            )
        if not isinstance(data["summary"],str):
            raise ValueError(
                "summary should be a string"
            )
        if not isinstance(data["keywords"],list):
            raise ValueError(
                "keywords must be a list"
            )
        return True;