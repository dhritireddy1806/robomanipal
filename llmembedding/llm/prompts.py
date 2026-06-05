from pathlib import Path
class PromptManager:
    def build_prompt(self,template_name,user_input):
        path=Path("templates")/template_name

        with open(path,"r") as f:
            template=f.read()

        return template.replace(
            "{user_input}",user_input
        )