from datetime import datetime
class Logger:
    def log_prompt(self , prompt):
        with open("logs.txt","a") as f:
            f.write("\n")
            f.write("="*50+"\n")
            f.write(f"Time:{datetime.now()}\n")
            f.write(prompt+"\n")

    def log_response(self,response):
        with open("logs.txt","a") as f:
            f.write("\n")
            f.write("Raw Response:\n")
            f.write(response+"\n")

    def log_error(self,error):
        with open("logs.txt","a") as f:
            f.write("\n")
            f.write("Error:\n")
            f.write(str(error)+"\n")