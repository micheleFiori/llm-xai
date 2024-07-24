from .XAILLM import XAILLM
import re

class RatingStrategy(XAILLM):

    def __init__(self, llm, csv, window_length, window_length_unit):
        systemprompt = f"""
Act as a system that has to evaluate the best explanation for a sensor-based activity recognition model in smart-homes.
You will see multiple explanations for the same time window.
The observed time window is {window_length} {window_length_unit} long.
You will have to provide a score (1 to 5) for each explanation considering a target user that is not expert in machine learning and activity recognition.
In order to do that you should examine each element in the explanation and consider if it can even partially explain the occurrence of the predicted activity
according to a common-sense knowledge or, on the contrary, it is unrelated to the particular predicted activity.
You should also consider that a good explanation should be clear and straightforward and should not include additional elements that
are not consistent with the predicted activity.
At the end, always provide the summarized answer for each explanation in this format: "OPTION=x,SCORE=y" where x is the number of the option while y is the score.
"""
# Reason step by step, analyzing each explanation separately.
        super().__init__(llm,csv,systemprompt.replace("\n",""))

    def parseResponse(self, response):
        regex_pattern = r'OPTION\s*=\s*(\d+)\s*,\s*SCORE\s*=\s*(\d+)'
        matches = re.findall(regex_pattern, response)
        scores = []
        for option, score in matches:
            scores.append({"option":int(option)-1,"score":int(score)})
        #return scores
        return [dict(t) for t in {tuple(d.items()) for d in scores}]


    def computeOutcome(self, scores):
        outcome = {"MP":0, "LIME":0, "GRADCAM":0}
        for i in range(len(scores)):
            for s in scores[i]:
                option = s["option"]
                score = s["score"]
                methods = self.questions[i][option]["methods"]
                for method in methods:
                    outcome[method]+=score
        return outcome