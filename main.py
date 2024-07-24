import argparse
from strategies.BestExplanationStrategy import BestExplanationStrategy
from strategies.RatingStrategy import RatingStrategy
from langchain_openai import ChatOpenAI
import os
import pickle

OPEN_AI_KEY = "YOUR-API-KEY"

def pickle_me(structure, dst):
    with open(dst, 'wb') as file:
        pickle.dump(structure , file)
    print("Data structure pickled successfully!")
    
if __name__ == "__main__":
    
    allowed_datasets = ['marble', 'casas']
    allowed_strategies = ['best', 'score']
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=allowed_datasets,  help=f"The dataset to use. Must be in {allowed_datasets}")
    parser.add_argument("strategy", type=str, choices=allowed_strategies, help=f"The evaluation strategy to use. Must be in {allowed_strategies}")
    parser.add_argument("model", type=str, nargs='?', default="gpt-3.5-turbo-0125", help=f"The model to use. Default is gpt-3.5-turbo-0125")
    parser.add_argument("temperature", type=float, nargs='?', default=0.0, help=f"The temperature to set. Default is 0")
    parser.add_argument("n_repetitions", type=int, nargs='?', default=1, help=f"The number of repetition for the experiments. Default is 1")
    args = parser.parse_args()
    
    if args.dataset=="marble":
        csv_path = f"data/marble.csv"
        window_len = "sixteen"
        window_unit = "seconds"
    elif args.dataset=="casas":
        csv_path = f"data/casas.csv"
        window_len = "six"
        window_unit = "minutes"
        
    llm = ChatOpenAI(model=args.model, temperature=args.temperature, openai_api_key=OPEN_AI_KEY)
    
    if args.strategy=="best":
        strategy = BestExplanationStrategy(llm, csv_path, window_len, window_unit)
    elif args.strategy=="score":
        strategy = RatingStrategy(llm, csv_path, window_len, window_unit)
           
                                           
    outcomes = []
    for i in range(args.n_repetitions):
        outcome = strategy.runExperiment()
        outcomes.append(outcome)
                                           
    result_folder = f"results/{args.dataset}/{args.model}/{args.strategy}"
    file_dest = f"{result_folder}/{args.model}_{args.n_repetitions}.pkl"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    pickle_me(outcomes, file_dest)