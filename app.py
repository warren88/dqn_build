import train
import evaluate
import agent
import sys

def main():
    while True:
        print("\nWelcome to the trading bot.")
        print("1. Train the model")
        print("2. Evaluate the model")
        print("3. Exit")
        user_input = input("Enter your choice: ")

        if user_input == "1":
            print("\nTraining the model...")
            agent = train.run_training()  # assuming your train.py has a run_training function
            print("Training completed.")

        elif user_input == "2":
            if 'agent' not in locals():
                print("\nPlease train the model first.")
                continue
            print("\nEvaluating the model...")
            evaluate.run_evaluation(agent)  # assuming your evaluate.py has a run_evaluation function
            print("Evaluation completed.")

        elif user_input == "3":
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid input. Please try again.")


if __name__ == "__main__":
    main()
