import pandas as pd
from Bio import SeqIO
import gzip
import ftplib
import os
import random

def download_fasta(ftp_url):
    """
    Download RNA sequences from FTP server.
    """
    try:
        url_parts = ftp_url.split("/")
        host = url_parts[2]
        filepath = "/".join(url_parts[3:])
        
        ftp = ftplib.FTP(host)
        ftp.login()

        local_filename = os.path.basename(filepath)
        with open(local_filename, 'wb') as f:
            ftp.retrbinary(f"RETR {filepath}", f.write)

        ftp.quit()
        return local_filename
    except Exception as e:
        print(f"Failed to download {ftp_url}: {e}")
        return None

def process_sequences(input_file, output_file):
    """
    Process RNA sequences and save to CSV.
    """
    df = pd.read_csv(input_file, sep="\t")
    
    with open(output_file, 'w') as out_csv:
        out_csv.write("seq,organism_part\n")

        for index, row in df.iterrows():
            ftp_url_1 = row['Comment[FASTQ_URI]']
            ftp_url_2 = ftp_url_1.replace("_1.", "_2.")

            for ftp_url in [ftp_url_1, ftp_url_2]:
                local_file = download_fasta(ftp_url)
                
                if local_file:
                    sequences = []
                    
                    with gzip.open(local_file, 'rt') as f:
                        for record in SeqIO.parse(f, "fastq"):
                            sequences.append(str(record.seq))

                    selected_sequences = random.sample(sequences, min(10, len(sequences)))
                    
                    for seq in selected_sequences:
                        out_csv.write(f"{seq},{row['Characteristics[genotype]']}\n")
                    
                    os.remove(local_file)

def main():
    input_file = 'data/E-MTAB-5530.sdrf.txt'
    output_file = 'data/dna_sequences.csv'
    process_sequences(input_file, output_file)
    print(f"DNA sequences and organism parts saved to {output_file}.")

if __name__ == "__main__":
    main()
