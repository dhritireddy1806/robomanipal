import json
class Parser:
    def parse(self,response):
        try:
            return json.loads(response)# coverts text into dictionary
        
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON response recieved from model:{e}"
            )
        