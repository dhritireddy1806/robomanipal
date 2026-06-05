import time
class Retry:
    def __init__(self,max_retry=3, delay=1):
        self.max_retry=max_retry
        self.delay=delay

    def execute(self,func,*args,**kwargs):
        le=None
        for a in range(1, self.max_retry+1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                le=e
                print(f"Attempt {a} failed:{e}")

                if a<self.max_retry:
                    time.sleep(self.delay)

        raise RuntimeError(
        "request failed after",self.max_retry,"attempts"
           )from le