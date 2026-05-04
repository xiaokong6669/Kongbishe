import sys

def process_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line.startswith('>'):  # 标题行
                outfile.write(line + '\n')
            else:  # 序列行
                padded_seq = line.ljust(50, 'X')[:50]  # 填充到50个字符，如果超过则截断
                outfile.write(padded_seq + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extend_fasta.py input.fasta output.fasta")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_fasta(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")