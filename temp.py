import re
from collections import defaultdict
import pandas as pd

def parse_batch_log(file_path):
    # Dictionary to store accuracies for each parameter set
    accuracies = defaultdict(lambda: {'top1': [], 'top3': []})
    current_param = None
    
    # Regular expressions to match accuracy lines
    top1_pattern = re.compile(r'Top-1 Accuracy on the test dataset: (\d+\.\d+)%')
    top3_pattern = re.compile(r'Top-3 Accuracy on the test dataset: (\d+\.\d+)%')
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Identify parameter set
        if line.startswith('Processing parameter:'):
            current_param = line.split('Processing parameter: ')[1].strip()
        
        # Extract Top-1 Accuracy
        top1_match = top1_pattern.search(line)
        if top1_match and current_param:
            accuracies[current_param]['top1'].append(float(top1_match.group(1)))
        
        # Extract Top-3 Accuracy
        top3_match = top3_pattern.search(line)
        if top3_match and current_param:
            accuracies[current_param]['top3'].append(float(top3_match.group(1)))
    
    return accuracies

def calculate_averages(accuracies):
    # Calculate average Top-1 and Top-3 accuracies for each parameter set
    results = []
    for param, acc in accuracies.items():
        avg_top1 = sum(acc['top1']) / len(acc['top1']) if acc['top1'] else 0
        avg_top3 = sum(acc['top3']) / len(acc['top3']) if acc['top3'] else 0
        results.append({
            'Parameter': param,
            'Avg Top-1 Accuracy (%)': round(avg_top1, 2),
            'Avg Top-3 Accuracy (%)': round(avg_top3, 2)
        })
    return results

def main():
    file_path = 'batch_output.log'
    # Parse the log file
    accuracies = parse_batch_log(file_path)
    
    # Calculate averages
    results = calculate_averages(accuracies)
    
    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(results)
    
    # Print results
    print("\nAverage Top-1 and Top-3 Accuracies per Parameter Set:")
    print(df.to_string(index=False))
    
    # Optionally save to CSV
    df.to_csv('average_accuracies.csv', index=False)
    print("\nResults saved to 'average_accuracies.csv'")

if __name__ == "__main__":
    main()
    print("Batch processing completed.")