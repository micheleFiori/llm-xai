from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
import csv
import tqdm
import time

class XAILLM(ABC):

    def __init__(self,llm, csv_file, systemprompt):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    systemprompt,
                ),
                ("human", "{input}"),
            ]
        )
        self.llm=llm
        #self.llm_chain = prompt | llm
        self.questions = self.extrapolateQuestions(csv_file)

    def extrapolateQuestions(self, csv_file):
        questions = []
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                question = []
                mp = row['MP']
                lime = row['LIME']
                gradcam = row['GRADCAM']
                if mp==lime:
                    question.append({"option": mp, "methods": ["MP","LIME"]})
                    question.append({"option": gradcam, "methods": ["GRADCAM"]})
                elif lime==gradcam:
                    question.append({"option": mp, "methods": ["MP"]})
                    question.append({"option": lime, "methods": ["LIME","GRADCAM"]})
                elif mp==gradcam:
                    question.append({"option": mp, "methods": ["MP","GRADCAM"]})
                    question.append({"option": lime, "methods": ["LIME"]})
                else:
                    question.append({"option": mp, "methods": ["MP"]})
                    question.append({"option": lime, "methods": ["LIME"]})
                    question.append({"option": gradcam, "methods": ["GRADCAM"]})

                questions.append(question)

        return questions

    def runExperiment(self):
        rawResponses = []
        parsedResponses = []
        for question in tqdm.tqdm(self.questions):
            time.sleep(0.01)
            user_prompt = self.createOptions(question)
            llm_chain = self.prompt | self.llm
            print(user_prompt)
            response = llm_chain.invoke({"input": user_prompt}).content
            print(response)
            rawResponses.append(response)
            parsedResponses.append(self.parseResponse(response))
        outcome = self.computeOutcome(parsedResponses)
        return {"raw": rawResponses, "parsed": parsedResponses, "outcome": outcome}


    def createOptions(self, question):
        res = ""
        for i in range(len(question)):
            res += str(i+1) + ")" + question[i]["option"] + '\n'
        return res.rstrip('\n')

    @abstractmethod
    def parseResponse(response):
        pass

    @abstractmethod
    def computeOutcome(response):
        pass