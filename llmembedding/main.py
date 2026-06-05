from llm.prompts import PromptManager
from llm.client import Client
from llm.parser import Parser
from llm.validator import Validator
from llm.logger import Logger
from llm.retry import Retry

logger=Logger()

try:

    retry=Retry()

    pm=PromptManager()
    prompt=pm.build_prompt(
    "summary_prompt.txt",
    "Respond however you want."
    )

    logger.log_prompt(prompt)

    client=Client()
    response=retry.execute(client.generate, prompt)
    print(response)
    logger.log_response(response)

    print('*'*60,'\n')
    print("After Parsing")
    parser=Parser()
    data=parser.parse(response)

    validator=Validator()
    validator.validate(data)
    print('*'*60,'\n')
    print("After validation")
    print(data)

except Exception as e:
    logger.log_error(e)
    print(e)