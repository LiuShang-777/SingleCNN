#this file receive correctly predicted gene names and total fasta file to extract true positive and true negative fasta 
import sys
input_file_name=sys.argv[1]
input_file_seq=sys.argv[2]
output_file=sys.argv[3]
with open(input_file_name,'r') as file:
    list_name=[]
    for i in file:
        i=i.strip()
        list_name.append(i)
with open(input_file_seq,'r') as file:
    list_id=[]
    list_seq=[]
    for line in file:
        line=line.strip()
        if line[0]=='>':
            list_id.append(line)
        else:
            list_seq.append(line)
dic={}
for i,j in zip(list_id,list_seq):
    dic[i]=j
with open(output_file,'w') as file:
    for i in list_name:
        file.write(i+'\n')
        file.write(dic[i]+'\n')

