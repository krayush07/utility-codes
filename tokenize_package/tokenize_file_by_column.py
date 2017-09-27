# -*- coding: utf-8 -*-
import nltk

def tokenize_by_column(filename, column_num=0):
    lines = open(filename, 'r')
    op_file = open(filename + '_tokenized', 'w')

    for line in lines:
        line_split = line.strip().split('\t')
        print line
        initial_string = '\t'.join(line_split[0:column_num]).strip()
        tokenized_string = ' '.join(nltk.word_tokenize(line_split[column_num].decode('utf-8'))).strip()
        last_string = '\t'.join(line_split[column_num+1:]).strip()
        final_string = initial_string + '\t' + tokenized_string.encode('utf-8') + '\t' + last_string
        op_file.write(final_string.strip() + '\n')

    op_file.close()

tokenize_by_column('/path', 2)