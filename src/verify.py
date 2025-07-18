from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse
import os


def  main():
    parser = argparse.ArgumentParser(
        description='Label images using a vision LLM and a text classifier.'
    )
    parser.add_argument(
        "--label_list",
        type=str,
        required=True,
        help="Directory to save prediction label.",
    )
    parser.add_argument(
        "--labels_shuffled",
        type=str,
        required=True,
        help="Directory to save original label",
    )
    args = parser.parse_args()

    # Load label_list
    if not os.path.exists(args.label_list):
        raise FileNotFoundError(f"Classes file {args.label_list} not found.")
    else:
        with open(args.label_list, "r") as f:
            label_prediction = [line.strip() for line in f.readlines()]
    if not label_prediction:
        raise ValueError("No classes found in the classes file.")
    
    # Load label_shuffled
    if not os.path.exists(args.labels_shuffled):
        raise FileNotFoundError(f"Classes file {args.labels_shuffled} not found.")
    else:
        with open(args.labels_shuffled, "r") as f:
            label_original= [line.strip() for line in f.readlines()]
    if not label_original:
        raise ValueError("No classes found in the classes file.")
    
    accuracy = accuracy_score(label_original, label_prediction)
    precision = precision_score(label_original, label_prediction, pos_label="目标区域") 
    recall = recall_score(label_original, label_prediction, pos_label="目标区域")
    print(f"Acc: {accuracy},Pre:{precision},Re:{recall}")
    
if __name__ == '__main__':
    main()