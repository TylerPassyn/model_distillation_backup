import pandas as pd 
import re 
import os 
import sys





def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python Extracting_Labels.py <input_file> <directory>")
        sys.exit(1)
    if not os.path.exists(args[0]):
        print(f"Error: The file {args[0]} does not exist.")
        sys.exit(1)
    if not os.path.isfile(args[0]):
        print(f"Error: The path {args[0]} is not a file.")
        sys.exit(1)
    if not os.path.exists(args[1]):
        print(f"Error: The directory {args[1]} does not exist.")
        sys.exit(1)
    if not os.path.isdir(args[1]):
        print(f"Error: The path {args[1]} is not a directory.")
        sys.exit(1)

    input_file = args[0]
    output_directory = args[1]
    output_file = os.path.join(output_directory, 'extracted_labels.csv')
    
        
    uncleaned_labels = []
    with open(input_file, 'r') as file:
        for line in file:
            #get everything between ** and **
            matches = re.findall(r'\*\*(.*?)\*\*', line)
            if matches:
                uncleaned_labels.extend(matches)

    #replace everything after the final after the first space with an empty string
    cleaned_labels = [label.split(' ')[0] for label in uncleaned_labels]

    #change _ to  a space
    cleaned_labels = [label.replace('_', ' ') for label in cleaned_labels]

    #convert the array to a pandas dataframe
    df = pd.DataFrame(cleaned_labels, columns=['Class Label'])
    df.to_csv(output_file, index=False, header = True, mode='w', encoding='utf-8', sep=',')
    print(f"Cleaned labels saved to {output_file}")


if __name__ == "__main__":
    main()


        
    


    
    
    
    
    



