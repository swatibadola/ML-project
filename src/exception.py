# for exception handling
# This code will be same for the entire project, wherever the try catch error will be used, custom exception will be called

import sys
import logging

def error_message_detail(error, error_detail:sys):
    # exc_tb => exception tab
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) 
        # super => to inherit the init 
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/00
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e, sys)