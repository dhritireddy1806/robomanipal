1. How prompt logic is separated from code
A. There is a sseparate template.txt file for the prompts and it is accessed by the main,py file so any changes to the input doesnt affect the client, partser or validation.py file 

2. How output validity is enforced
A. Its done in 2 stages parsing and validation. PArsing is done to ensre that the response is valid json. Validation checks if output follows the expected scheme

3. How failures are handled
A. Retry.py handles the failure 
if the caode fails this is the procedure
Attempt 1-fails
Attempt 2-fails
Attempt 3- fails
Attempt 4-Success
if it fails even after all the number of retries available then an exception is raised and error is logged and error is displayed

4. One limitation of your current design
A. Each time i want to run the code using the different possibilities like for experimenting error, json format , without json format or proper functioning i need to make chcnges in the same template which is not conventional.