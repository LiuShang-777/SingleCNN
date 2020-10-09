# this file shuffle extracted genes sequences(receive fasta and output fasta)
import sys
import random
input_file=sys.argv[1]
output_file=sys.argv[2]
def get_list(input_file):
    with open(input_file,'r') as file:
        list_id,list_seq=[],[]
        for line in file:
            line=line.strip()
            if line[0]=='>':
                list_id.append(line)
            else:
                list_seq.append(line)
    return (list_id,list_seq)
list_id,list_seq=get_list(input_file)
def sishuffle(list_id,list_seq):
    list_shuffle=[]
    for i,j in zip(list_id,list_seq):
        chrlist=list(j)
        random.shuffle(chrlist)
        list_shuffle.append(''.join(chrlist))
    return (list_id,list_shuffle)
list_id,list_new_seq=sishuffle(list_id,list_seq)
with open(output_file,'w') as file:
    for i,j in zip(list_id,list_new_seq):
        file.write(i+'\n')
        file.write(j+'\n')
