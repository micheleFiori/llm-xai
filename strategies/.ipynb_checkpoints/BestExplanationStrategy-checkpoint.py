from .XAILLM import XAILLM
import re

class BestExplanationStrategy(XAILLM):

    def __init__(self, llm, csv, window_length, window_length_unit):
        systemprompt = f"""
Act as a system that has to evaluate the best explanation for a sensor-based activity recognition model in smart-homes.
You will see multiple explanations for the same time window.
The observed time window is {window_length} {window_length_unit} long.
You will have to choose the explanation that is best for a target user that is not expert in machine learning and activity recognition.
In order to do that you should examine each element in the explanation and consider if it can even partially explain the occurrence of the predicted activity
according to a common-sense knowledge or, on the contrary, it is unrelated to the particular predicted activity.
You should also consider that a good explanation should be clear and straightforward and should not include additional elements that
are not consistent with the predicted activity.
Please at the end provide the answer in this format: "CHOICE=x" where x is the number of the option you chose.
"""
#Reason step by step, analyzing each explanation separately.
        super().__init__(llm,csv, systemprompt.replace("\n",""))

    def parseResponse(self, response):
        try:
          choice_number = re.search(r'CHOICE[=:]\s*(\d+)', response, flags=re.IGNORECASE).group(1)
          return int(choice_number)-1
        except AttributeError:
          print("NOT PARSED")


    def computeOutcome(self, responses):
        outcome = {"MP":0, "LIME":0, "GRADCAM":0}
        for i in range(len(responses)): #For each response from gpt
            winningMethods = self.questions[i][responses[i]]["methods"] #Remember that sometimes two XAI methods provide same explanation
            for method in winningMethods:
                outcome[method]+=1
        return outcome